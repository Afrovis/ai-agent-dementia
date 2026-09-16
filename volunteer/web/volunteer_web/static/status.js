// /s/{id}#k=...&d=... (HANDOFF.md section 5.4 and 5.5). The fragment never
// leaves the browser: it is read here and used only for local decryption
// and the delete call's body.

(function () {
  "use strict";

  const crypto_ = window.VolunteerCrypto;
  const POLL_INTERVAL_MS = 4000;

  const els = {
    statusText: document.getElementById("status-text"),
    downloadSection: document.getElementById("download-section"),
    downloadLink: document.getElementById("download-link"),
    deleteButton: document.getElementById("delete-button"),
    deleteConfirmed: document.getElementById("delete-confirmed"),
    errorText: document.getElementById("error-text"),
  };

  function b64urlDecode(value) {
    const padded =
      value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (value.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function parseHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    return { k: params.get("k"), d: params.get("d") };
  }

  function submissionIdFromPath() {
    const match = location.pathname.match(/\/s\/([^/]+)/);
    return match ? match[1] : "";
  }

  async function poll(submissionId, k) {
    const response = await fetch(`/api/submissions/${submissionId}`);
    if (!response.ok) {
      els.errorText.textContent = "This link doesn't match a known submission.";
      return;
    }
    const body = await response.json();
    if (body.status === "ready") {
      els.statusText.textContent = "Ready. Decrypting your video in your browser...";
      await downloadResult(submissionId, k);
      return;
    }
    if (body.status === "failed") {
      els.statusText.textContent = body.error_public || "Processing failed.";
      return;
    }
    if (body.status === "deleted") {
      els.statusText.textContent = "This video has been deleted.";
      return;
    }
    const position = body.queue_position ? ` (position ${body.queue_position} in queue)` : "";
    els.statusText.textContent = `Status: ${body.status}${position}. This page updates automatically.`;
    setTimeout(() => poll(submissionId, k), POLL_INTERVAL_MS);
  }

  async function downloadResult(submissionId, k) {
    const response = await fetch(`/api/submissions/${submissionId}/result`);
    if (!response.ok) {
      els.errorText.textContent = "Could not fetch the result yet, please retry shortly.";
      setTimeout(() => poll(submissionId, k), POLL_INTERVAL_MS);
      return;
    }
    const encrypted = new Uint8Array(await response.arrayBuffer());
    const key = b64urlDecode(k);
    const plaintext = await crypto_.decryptResultStream(key, submissionId, encrypted);
    const blob = new Blob([plaintext], { type: "video/mp4" });
    const url = URL.createObjectURL(blob);
    els.downloadLink.href = url;
    els.downloadLink.download = "analysis.mp4";
    els.downloadSection.hidden = false;
    els.statusText.textContent = "Your analysis video is ready.";
  }

  function main() {
    const submissionId = submissionIdFromPath();
    const { k, d } = parseHash();
    if (!submissionId || !k || !d) {
      els.errorText.textContent =
        "This link is missing part of its address. Use the exact link you were given after recording.";
      return;
    }

    poll(submissionId, k);

    els.deleteButton.addEventListener("click", async () => {
      if (!confirm("Delete your recording and its analysis? This cannot be undone.")) return;
      const response = await fetch(`/api/submissions/${submissionId}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ deletion_token: d }),
      });
      if (response.ok) {
        els.deleteConfirmed.hidden = false;
        els.statusText.textContent = "Deleted.";
        els.downloadSection.hidden = true;
      } else {
        els.errorText.textContent = "Could not delete right now, please try again.";
      }
    });
  }

  main();
})();
