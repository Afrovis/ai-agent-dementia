import json

from light.backends import DisabledBackend, ShellyBackend


def test_disabled_backend_records_absolute_states():
    backend = DisabledBackend()
    assert backend.set_state(True)
    assert backend.set_state(False)
    assert backend.states == [True, False]


def test_shelly_backend_calls_local_switch_set():
    calls = []

    class Response:
        status = 200

    def open_request(request, **kwargs):
        calls.append((request, kwargs))
        return Response()

    backend = ShellyBackend("http://192.0.2.10/", switch_id=1, open_fn=open_request)
    assert backend.set_state(True)
    request, kwargs = calls[0]
    assert request.full_url == "http://192.0.2.10/rpc/Switch.Set"
    assert request.method == "POST"
    assert json.loads(request.data) == {"id": 1, "on": True}
    assert kwargs == {"timeout": 3.0}


def test_shelly_backend_reports_http_failure():
    class Response:
        status = 503

    def open_request(_request, **_kwargs):
        return Response()

    assert not ShellyBackend("http://plug", open_fn=open_request).set_state(False)
