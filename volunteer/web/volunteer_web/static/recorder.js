// Consent -> setup -> guided recording flow for /record (PLAN.md "Volunteer
// flow", HANDOFF.md sections 5.6-5.7). Plain script, no bundler: relies on
// window.VolunteerCrypto from crypto.js, loaded first in record.html.

(function () {
  "use strict";

  const crypto_ = window.VolunteerCrypto;
  const MAX_CHUNK_RETRIES = 3;

  const els = {
    consentSection: document.getElementById("consent-section"),
    setupSection: document.getElementById("setup-section"),
    recordingSection: document.getElementById("recording-section"),
    doneSection: document.getElementById("done-section"),
    unsupportedSection: document.getElementById("unsupported-section"),
    turnstileContainer: document.getElementById("turnstile-container"),
    consentAdult: document.getElementById("consent-adult"),
    consentUnderstood: document.getElementById("consent-understood"),
    consentAlone: document.getElementById("consent-alone"),
    consentSubmit: document.getElementById("consent-submit"),
    bookmarkLink: document.getElementById("bookmark-link"),
    setupPreview: document.getElementById("setup-preview"),
    testPromptsButton: document.getElementById("test-prompts-button"),
    startRecordingButton: document.getElementById("start-recording-button"),
    promptText: document.getElementById("prompt-text"),
    promptSafety: document.getElementById("prompt-safety"),
    skipStepButton: document.getElementById("skip-step-button"),
    stopButton: document.getElementById("stop-button"),
    pendingChunks: document.getElementById("pending-chunks"),
    doneMessage: document.getElementById("done-message"),
  };

  function supportsRecording() {
    return Boolean(
      window.crypto &&
        window.crypto.subtle &&
        window.MediaRecorder &&
        navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia,
    );
  }

  function show(section) {
    for (const el of [
      els.consentSection,
      els.setupSection,
      els.recordingSection,
      els.doneSection,
      els.unsupportedSection,
    ]) {
      if (el) el.hidden = el !== section;
    }
  }

  function pickMimeType() {
    const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/mp4"];
    return candidates.find((type) => window.MediaRecorder.isTypeSupported(type)) || "";
  }

  async function sha256Hex(bytes) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  function b64urlEncode(bytes) {
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function b64urlDecode(value) {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (value.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function playPromptAudio(stepId) {
    return new Promise((resolve) => {
      const audio = new Audio(`/prompts/${stepId}.wav`);
      let settled = false;
      const done = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };
      audio.addEventListener("ended", done);
      audio.addEventListener("error", () => {
        // Prompt audio isn't rendered yet (V10) -- fall back to a plain
        // beep. Never speechSynthesis (HANDOFF.md rule: some browser
        // voices are cloud-backed).
        beep().then(done);
      });
      audio.play().catch(() => beep().then(done));
    });
  }

  async function beep() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const oscillator = ctx.createOscillator();
    oscillator.frequency.value = 880;
    oscillator.connect(ctx.destination);
    oscillator.start();
    await sleep(400);
    oscillator.stop();
    await ctx.close();
  }

  async function main() {
    if (!supportsRecording()) {
      show(els.unsupportedSection);
      return;
    }

    const configResponse = await fetch("/api/config");
    const config = await configResponse.json();

    let turnstileToken = "";
    window.onloadTurnstileCallback = function () {
      window.turnstile.render(els.turnstileContainer, {
        sitekey: config.turnstile_site_key,
        appearance: "interaction-only",
        execution: "render",
        action: "submit",
        callback: (token) => {
          turnstileToken = token;
          updateConsentButton();
        },
        "expired-callback": () => {
          turnstileToken = "";
          updateConsentButton();
        },
        "error-callback": () => {
          turnstileToken = "";
          updateConsentButton();
        },
      });
    };
    const turnstileScript = document.createElement("script");
    turnstileScript.src =
      "https://challenges.cloudflare.com/turnstile/v0/api.js?onload=onloadTurnstileCallback&render=explicit";
    turnstileScript.async = true;
    document.head.appendChild(turnstileScript);

    function updateConsentButton() {
      const allChecked =
        els.consentAdult.checked && els.consentUnderstood.checked && els.consentAlone.checked;
      els.consentSubmit.disabled = !(allChecked && turnstileToken);
    }
    for (const box of [els.consentAdult, els.consentUnderstood, els.consentAlone]) {
      box.addEventListener("change", updateConsentButton);
    }

    let submissionId = "";
    let uploadToken = "";
    let aesKey = null;

    els.consentSubmit.addEventListener("click", async () => {
      els.consentSubmit.disabled = true;
      const secrets = crypto_.generateSessionSecrets();
      const publicKeyDer = b64urlDecode(config.public_key_spki);
      const publicKey = await crypto_.importPublicKeySpki(publicKeyDer);
      const wrappedKey = await crypto_.wrapKey(publicKey, secrets.k);
      const deletionTokenSha256 = await sha256Hex(secrets.d);

      const response = await fetch("/api/submissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turnstile_token: turnstileToken,
          wrapped_key: b64urlEncode(wrappedKey),
          deletion_token_sha256: deletionTokenSha256,
          consent_version: config.consent_version,
          consents: { adult: true, understood: true, alone: true },
        }),
      });
      if (!response.ok) {
        els.consentSubmit.disabled = false;
        alert("Something went wrong starting your session. Please try again.");
        return;
      }
      const created = await response.json();
      submissionId = created.id;
      uploadToken = created.upload_token;
      aesKey = secrets.k;

      const statusUrl = `${location.origin}/s/${submissionId}#k=${b64urlEncode(secrets.k)}&d=${b64urlEncode(secrets.d)}`;
      els.bookmarkLink.href = statusUrl;
      els.bookmarkLink.textContent = statusUrl;

      await startSetup();
    });

    let previewStream = null;

    async function startSetup() {
      show(els.setupSection);
      previewStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      els.setupPreview.srcObject = previewStream;
    }

    els.testPromptsButton.addEventListener("click", () => {
      playPromptAudio("stand_still");
    });

    els.startRecordingButton.addEventListener("click", () => {
      startRecording();
    });

    async function startRecording() {
      show(els.recordingSection);
      const scriptResponse = await fetch("/script.json");
      const steps = await scriptResponse.json();

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(previewStream, {
        mimeType,
        videoBitsPerSecond: 2_500_000,
      });

      let chunkIndex = 0;
      let pending = 0;
      let stopped = false;
      const markers = [];
      const startTime = performance.now();

      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size === 0) return;
        const n = chunkIndex;
        chunkIndex += 1;
        pending += 1;
        updatePendingUi(pending);
        uploadChunk(event.data, n).finally(() => {
          pending -= 1;
          updatePendingUi(pending);
        });
      });

      function updatePendingUi(count) {
        if (els.pendingChunks) els.pendingChunks.textContent = String(count);
      }

      async function uploadChunk(blob, n) {
        const plaintext = new Uint8Array(await blob.arrayBuffer());
        const iv = crypto.getRandomValues(new Uint8Array(crypto_.GCM_IV_BYTES));
        const encrypted = await crypto_.encryptChunk(aesKey, submissionId, n, plaintext, iv);
        for (let attempt = 0; attempt < MAX_CHUNK_RETRIES; attempt += 1) {
          try {
            const response = await fetch(`/api/submissions/${submissionId}/chunks/${n}`, {
              method: "PUT",
              headers: {
                Authorization: `Bearer ${uploadToken}`,
                "Content-Type": "application/octet-stream",
              },
              body: encrypted,
            });
            if (response.ok) return;
          } catch {
            // fall through to retry
          }
          await sleep(500 * 2 ** attempt);
        }
      }

      recorder.start(5000);

      const hardCapTimer = setTimeout(() => {
        if (!stopped) stopRecording();
      }, config.max_recording_seconds * 1000);

      els.stopButton.addEventListener("click", () => stopRecording());

      async function runSteps() {
        for (const step of steps) {
          if (stopped) return;
          els.promptText.textContent = step.text;
          els.promptSafety.textContent = step.safety_note || "";
          els.skipStepButton.hidden = !step.optional;
          await playPromptAudio(step.step_id);
          const tStart = (performance.now() - startTime) / 1000;

          let skipped = false;
          await Promise.race([
            sleep(step.seconds * 1000),
            new Promise((resolve) => {
              els.skipStepButton.onclick = () => {
                skipped = true;
                resolve();
              };
            }),
          ]);
          els.skipStepButton.onclick = null;

          const tEnd = (performance.now() - startTime) / 1000;
          markers.push({
            step_id: step.step_id,
            expected_state: step.expected_state,
            t_start_s: tStart,
            t_end_s: tEnd,
            skipped,
          });
        }
        if (!stopped) stopRecording();
      }
      runSteps();

      async function stopRecording() {
        if (stopped) return;
        stopped = true;
        clearTimeout(hardCapTimer);
        const stoppedPromise = new Promise((resolve) => recorder.addEventListener("stop", resolve));
        recorder.stop();
        await stoppedPromise;
        for (const track of previewStream.getTracks()) track.stop();

        const duration_s = (performance.now() - startTime) / 1000;
        await fetch(`/api/submissions/${submissionId}/finalize`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${uploadToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            chunk_count: chunkIndex,
            duration_s,
            mime_type: mimeType,
            markers,
          }),
        });

        show(els.doneSection);
        const statusUrl = els.bookmarkLink.href;
        els.doneMessage.textContent = "Uploaded. Taking you to your results page...";
        setTimeout(() => {
          location.href = statusUrl;
        }, 2000);
      }
    }
  }

  main();
})();
