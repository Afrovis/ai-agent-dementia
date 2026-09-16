from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
from volunteer_common import db, paths

from volunteer_web.app import b64url_encode, create_app
from volunteer_web.config import WebConfig
from volunteer_web.turnstile import StubVerifier

VALID_CONSENTS = {"adult": True, "understood": True, "alone": True}


def make_client(tmp_path: Path, *, always_pass: bool = True, **config_overrides) -> TestClient:
    defaults = dict(
        data_dir=str(tmp_path),
        min_free_disk_mb=0,
        chunk_max_bytes=1024,
        max_upload_mb=1,
        submissions_per_ip_per_hour=5,
    )
    defaults.update(config_overrides)
    config = WebConfig(**defaults)
    app = create_app(config, turnstile_verifier=StubVerifier(always_pass=always_pass))
    return TestClient(app)


def create_submission(client: TestClient, **overrides) -> dict:
    body = {
        "turnstile_token": "tok",
        "wrapped_key": b64url_encode(b"wrapped-key-bytes"),
        "deletion_token_sha256": hashlib.sha256(b"deletion-token").hexdigest(),
        "consent_version": "v1",
        "consents": VALID_CONSENTS,
    }
    body.update(overrides)
    response = client.post("/api/submissions", json=body)
    return response


def test_create_submission_success(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = create_submission(client)
    assert response.status_code == 201
    payload = response.json()
    assert len(payload["id"]) == 22
    assert payload["upload_token"]
    assert response.headers["cache-control"] == "no-store"


def test_create_submission_rejects_missing_consent(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = create_submission(
        client, consents={"adult": True, "understood": False, "alone": True}
    )
    assert response.status_code == 400


def test_create_submission_turnstile_failure(tmp_path: Path) -> None:
    client = make_client(tmp_path, always_pass=False)
    response = create_submission(client)
    assert response.status_code == 403


def test_create_submission_rate_limited(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for _ in range(5):
        assert create_submission(client).status_code == 201
    assert create_submission(client).status_code == 429


def test_create_submission_disk_guard(tmp_path: Path) -> None:
    client = make_client(tmp_path, min_free_disk_mb=10**9)
    assert create_submission(client).status_code == 507


def test_put_chunk_requires_valid_token(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    submission_id = create_submission(client).json()["id"]
    response = client.put(f"/api/submissions/{submission_id}/chunks/0", content=b"data")
    assert response.status_code == 401


def test_put_chunk_success_and_overwrite(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    headers = {"Authorization": f"Bearer {created['upload_token']}"}

    response = client.put(
        f"/api/submissions/{created['id']}/chunks/0", content=b"hello", headers=headers
    )
    assert response.status_code == 204

    chunk_path = paths.chunk_path(tmp_path, created["id"], 0)
    assert chunk_path.read_bytes() == b"hello"

    # Re-PUT overwrites in place rather than appending or double-counting.
    response = client.put(
        f"/api/submissions/{created['id']}/chunks/0", content=b"hello world", headers=headers
    )
    assert response.status_code == 204
    assert chunk_path.read_bytes() == b"hello world"

    conn = db.connect(paths.db_path(tmp_path))
    row = db.get_submission(conn, created["id"])
    assert row["chunk_count"] == 1


def test_put_chunk_rejects_when_not_recording(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    headers = {"Authorization": f"Bearer {created['upload_token']}"}
    client.put(f"/api/submissions/{created['id']}/chunks/0", content=b"x", headers=headers)
    finalize = client.post(
        f"/api/submissions/{created['id']}/finalize",
        json={"chunk_count": 1, "duration_s": 1.0, "markers": []},
        headers=headers,
    )
    assert finalize.status_code == 204

    response = client.put(
        f"/api/submissions/{created['id']}/chunks/1", content=b"y", headers=headers
    )
    assert response.status_code == 409


def test_put_chunk_too_large(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    headers = {"Authorization": f"Bearer {created['upload_token']}"}
    response = client.put(
        f"/api/submissions/{created['id']}/chunks/0",
        content=b"x" * 2048,
        headers=headers,
    )
    assert response.status_code == 413


def test_finalize_missing_chunk(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    headers = {"Authorization": f"Bearer {created['upload_token']}"}
    client.put(f"/api/submissions/{created['id']}/chunks/0", content=b"x", headers=headers)
    response = client.post(
        f"/api/submissions/{created['id']}/finalize",
        json={"chunk_count": 2, "duration_s": 1.0, "markers": []},
        headers=headers,
    )
    assert response.status_code == 422


def test_finalize_wrong_token(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    response = client.post(
        f"/api/submissions/{created['id']}/finalize",
        json={"chunk_count": 0, "duration_s": 1.0, "markers": []},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_get_status_not_found(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/api/submissions/" + "a" * 22).status_code == 404


def test_get_status_ok(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    response = client.get(f"/api/submissions/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "recording"
    assert body["queue_position"] is None


def test_result_not_ready_until_status_is_ready(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    assert client.get(f"/api/submissions/{created['id']}/result").status_code == 404


def test_result_served_once_ready(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    conn = db.connect(paths.db_path(tmp_path))
    conn.execute("UPDATE submissions SET status = 'ready' WHERE id = ?", (created["id"],))
    conn.commit()
    result_path = paths.result_path(tmp_path, created["id"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b"encrypted-result-bytes")

    response = client.get(f"/api/submissions/{created['id']}/result")
    assert response.status_code == 200
    assert response.content == b"encrypted-result-bytes"


def test_delete_wrong_token_is_forbidden(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    response = client.post(
        f"/api/submissions/{created['id']}/delete", json={"deletion_token": "wrong"}
    )
    assert response.status_code == 403


def test_delete_correct_token_marks_deleted(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = create_submission(client)
    created = response.json()
    delete_response = client.post(
        f"/api/submissions/{created['id']}/delete", json={"deletion_token": "deletion-token"}
    )
    assert delete_response.status_code == 204

    conn = db.connect(paths.db_path(tmp_path))
    row = db.get_submission(conn, created["id"])
    assert row["status"] == "deleted"


def test_api_config_returns_public_key_when_present(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    from cryptography.hazmat.primitives import serialization

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (tmp_path / "public.pem").write_bytes(public_pem)

    client = make_client(tmp_path)
    response = client.get("/api/config")
    assert response.status_code == 200
    body = response.json()
    assert body["public_key_spki"]
    assert body["consent_version"] == "v1"
    assert response.headers["cache-control"] == "no-store"


def test_no_stray_files_written_during_full_flow(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    created = create_submission(client).json()
    headers = {"Authorization": f"Bearer {created['upload_token']}"}
    client.put(f"/api/submissions/{created['id']}/chunks/0", content=b"a", headers=headers)
    client.put(f"/api/submissions/{created['id']}/chunks/1", content=b"b", headers=headers)
    client.post(
        f"/api/submissions/{created['id']}/finalize",
        json={"chunk_count": 2, "duration_s": 2.0, "markers": []},
        headers=headers,
    )

    written = sorted(
        p.name for p in paths.incoming_dir(tmp_path, created["id"]).iterdir() if p.is_file()
    )
    assert written == ["chunk_000000.enc", "chunk_000001.enc", "key.wrapped", "submission.json"]
