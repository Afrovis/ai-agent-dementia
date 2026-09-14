// Night Companion embodiment page client.
//
// Connects to the `/ws` endpoint, applies `show`/`say` messages to the DOM,
// and reconnects with exponential backoff on disconnect. No framework, per
// HANDOFF.md section 4 ("Embodiment page is plain HTML, CSS, and JS").

(function () {
  "use strict";

  const faceEl = document.getElementById("face");
  const headlineEl = document.getElementById("headline");
  const bodyEl = document.getElementById("body-text");
  const photoEl = document.getElementById("photo");
  const photoImgEl = document.getElementById("photo-img");
  const brightnessOverlayEl = document.getElementById("brightness-overlay");

  const FACE_STATES = ["asleep", "awake", "speaking", "listening"];
  let currentSpeech = null;
  let currentSpeechInterruptible = false;
  let currentSpeechSessionId = null;

  // A photo the caregiver never uploaded (or has since deleted) answers 404.
  // Fade the layer back out and hide it rather than leaving the browser's
  // broken-image glyph on a bedroom wall at 3am.
  photoImgEl.addEventListener("error", () => {
    photoEl.classList.remove("visible");
    photoEl.classList.add("hidden");
  });

  function applyShow(msg) {
    if (msg.face && FACE_STATES.includes(msg.face)) {
      FACE_STATES.forEach((state) => faceEl.classList.remove(state));
      faceEl.classList.add(msg.face);
    }
    if (typeof msg.headline === "string") {
      headlineEl.textContent = msg.headline;
    }
    if (typeof msg.body === "string") {
      bodyEl.textContent = msg.body;
    }
    if (msg.photo_id) {
      photoImgEl.src = `/photos/${msg.photo_id}`;
      photoEl.classList.remove("hidden");
      // Force reflow so the opacity transition (fade-in) runs even if the
      // src did not change.
      requestAnimationFrame(() => photoEl.classList.add("visible"));
    } else {
      photoEl.classList.remove("visible");
    }
    if (typeof msg.brightness === "number") {
      const brightness = Math.min(1, Math.max(0, msg.brightness));
      brightnessOverlayEl.style.opacity = String(1 - brightness);
    }
  }

  function applySay(msg) {
    // The face briefly shows the "speaking" state while text-to-speech is
    // expected to be playing; the agent's next `show` event will restore
    // whatever state follows.
    faceEl.classList.remove(...FACE_STATES);
    faceEl.classList.add("speaking");
    if (typeof msg.text === "string") {
      bodyEl.textContent = msg.text;
    }
    if (typeof msg.audio_url === "string") {
      if (currentSpeech) {
        currentSpeech.pause();
      }
      const speech = new Audio(msg.audio_url);
      currentSpeech = speech;
      currentSpeechInterruptible = msg.interruptible === true;
      currentSpeechSessionId = msg.session_id || null;
      speech.addEventListener("ended", () => {
        // An older clip can finish after a newer Say replaced it. Do not let
        // that stale callback erase the newer clip's barge-in state.
        if (currentSpeech === speech) {
          currentSpeech = null;
          currentSpeechInterruptible = false;
          currentSpeechSessionId = null;
        }
      });
      speech.play().catch((err) => {
        console.error("Piper speech playback failed", err);
      });
    }
  }

  function applySpeechStarted(msg) {
    if (!currentSpeech || !currentSpeechInterruptible) {
      return;
    }
    if (msg.session_id && currentSpeechSessionId !== msg.session_id) {
      return;
    }
    currentSpeech.pause();
    currentSpeech.currentTime = 0;
    currentSpeech = null;
    currentSpeechInterruptible = false;
    currentSpeechSessionId = null;
    faceEl.classList.remove(...FACE_STATES);
    faceEl.classList.add("listening");
  }

  function handleMessage(event) {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error("could not parse websocket message", err);
      return;
    }
    if (msg.type === "show") {
      applyShow(msg);
    } else if (msg.type === "say") {
      applySay(msg);
    } else if (msg.type === "speech_started") {
      applySpeechStarted(msg);
    }
  }

  function wsUrl() {
    const isSecure = window.location.protocol === "https:";
    const scheme = isSecure ? "wss" : "ws";
    return `${scheme}://${window.location.host}/ws`;
  }

  function connect(backoffMs) {
    const currentBackoff = backoffMs || 1000;
    const socket = new WebSocket(wsUrl());

    socket.addEventListener("open", () => {
      console.log("embodiment websocket connected");
    });

    socket.addEventListener("message", handleMessage);

    socket.addEventListener("close", () => {
      const nextBackoff = Math.min(currentBackoff * 2, 30000);
      setTimeout(() => connect(nextBackoff), currentBackoff);
    });

    socket.addEventListener("error", () => {
      socket.close();
    });
  }

  connect(1000);

  // --- Browser media bridge (issue #28) ---------------------------------
  //
  // Captures the webcam (downscaled, 2 fps JPEG) and mic (16 kHz mono
  // PCM16) and streams both to the `/media` websocket as JSON text
  // messages, so `capture`/`listen` can treat this browser tab as one
  // source among others (HANDOFF.md section 5: `Frame.source_kind` includes
  // `"browser"`). This is a trusted bedside device, so permission is
  // requested on page load rather than behind a user gesture; if it is
  // denied, we log to the console and to the on-page status line and the
  // rest of the page keeps working with no media bridge.

  const FRAME_FPS = 2;
  const FRAME_WIDTH = 320;
  const FRAME_HEIGHT = 240;
  const FRAME_JPEG_QUALITY = 0.6;
  const AUDIO_SAMPLE_RATE = 16000;
  // 4096 samples at the browser's native audio rate (e.g. 48 kHz) is a
  // manageable buffer size for a ScriptProcessorNode and keeps the
  // resulting 16 kHz chunks well under a second each.
  const AUDIO_BUFFER_SIZE = 4096;

  const statusEl = document.getElementById("media-status");

  function setMediaStatus(text) {
    console.log(`media bridge: ${text}`);
    if (statusEl) {
      statusEl.textContent = text;
    }
  }

  function mediaWsUrl() {
    const isSecure = window.location.protocol === "https:";
    const scheme = isSecure ? "wss" : "ws";
    return `${scheme}://${window.location.host}/media`;
  }

  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        // reader.result is a data URL: "data:image/jpeg;base64,<data>".
        const commaIndex = reader.result.indexOf(",");
        resolve(reader.result.slice(commaIndex + 1));
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // Fit the whole camera image inside the fixed 4:3 transport canvas without
  // changing its proportions. A 16:9 stream becomes 320x180 at (0, 30).
  function letterboxRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);
    return {
      x: Math.floor((targetWidth - width) / 2),
      y: Math.floor((targetHeight - height) / 2),
      width,
      height,
    };
  }

  // Downmixes and resamples a `Float32Array` of native-rate audio samples
  // (one or more channels) to mono PCM16 at `AUDIO_SAMPLE_RATE`.
  function floatSamplesToPcm16(channelData, nativeSampleRate) {
    const ratio = nativeSampleRate / AUDIO_SAMPLE_RATE;
    const outLength = Math.floor(channelData.length / ratio);
    const pcm16 = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const sample = channelData[Math.floor(i * ratio)];
      const clamped = Math.max(-1, Math.min(1, sample));
      pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return pcm16;
  }

  // A single, stable handle to the current `/media` websocket. Capture
  // loops (webcam/mic) hold onto this one object for the page's whole
  // lifetime; reconnects just swap `link.socket`/`link.ready` underneath
  // them, mirroring the reconnect-with-backoff pattern already used for
  // `/ws` above.
  function createMediaLink() {
    const link = { socket: null, ready: false };

    function connect(backoffMs) {
      const currentBackoff = backoffMs || 1000;
      const socket = new WebSocket(mediaWsUrl());
      link.socket = socket;

      socket.addEventListener("open", () => {
        link.ready = true;
        setMediaStatus("media bridge connected");
      });

      socket.addEventListener("close", () => {
        link.ready = false;
        const nextBackoff = Math.min(currentBackoff * 2, 30000);
        setMediaStatus("media bridge disconnected, retrying");
        setTimeout(() => connect(nextBackoff), currentBackoff);
      });

      socket.addEventListener("error", () => {
        socket.close();
      });
    }

    connect(1000);

    link.send = (message) => {
      if (link.ready) {
        link.socket.send(JSON.stringify(message));
      }
    };
    return link;
  }

  function startWebcamCapture(stream, media) {
    const video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.play().catch((err) => console.error("webcam preview failed to start", err));

    const canvas = document.createElement("canvas");
    canvas.width = FRAME_WIDTH;
    canvas.height = FRAME_HEIGHT;
    const ctx = canvas.getContext("2d");

    setInterval(() => {
      if (video.readyState < video.HAVE_CURRENT_DATA) {
        return;
      }
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      if (sourceWidth <= 0 || sourceHeight <= 0) {
        return;
      }
      const destination = letterboxRect(
        sourceWidth,
        sourceHeight,
        FRAME_WIDTH,
        FRAME_HEIGHT
      );
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      ctx.drawImage(
        video,
        destination.x,
        destination.y,
        destination.width,
        destination.height
      );
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            return;
          }
          blobToBase64(blob).then((jpegB64) => {
            media.send({
              type: "frame",
              jpeg_b64: jpegB64,
              width: FRAME_WIDTH,
              height: FRAME_HEIGHT,
              source_width: sourceWidth,
              source_height: sourceHeight,
            });
          });
        },
        "image/jpeg",
        FRAME_JPEG_QUALITY
      );
    }, 1000 / FRAME_FPS);
  }

  function startMicCapture(stream, media) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const source = audioContext.createMediaStreamSource(stream);
    // ScriptProcessorNode is deprecated in favour of AudioWorklet, but is
    // used here deliberately: this is a dev/MVP bridge and the extra
    // AudioWorklet module-loading plumbing is not worth it yet. Revisit if
    // browsers drop support.
    const processor = audioContext.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);

    source.connect(processor);
    // A ScriptProcessorNode only fires its `audioprocess` event while
    // connected into the graph, so it must be connected to a destination;
    // gain 0 keeps the mic silently monitored without playing it back.
    const silence = audioContext.createGain();
    silence.gain.value = 0;
    processor.connect(silence);
    silence.connect(audioContext.destination);

    processor.addEventListener("audioprocess", (event) => {
      const channelData = event.inputBuffer.getChannelData(0);
      const pcm16 = floatSamplesToPcm16(channelData, audioContext.sampleRate);
      media.send({
        type: "audio",
        pcm16_b64: arrayBufferToBase64(pcm16.buffer),
        sample_rate: AUDIO_SAMPLE_RATE,
      });
    });
  }

  async function startMediaBridge() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMediaStatus("getUserMedia unavailable (needs HTTPS or localhost)");
      return;
    }
    let stream;
    try {
      // Browser/OS acoustic echo cancellation is the v1 barge-in boundary.
      // VAD sees this capture; Night Companion intentionally adds no software AEC.
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: { echoCancellation: true },
      });
    } catch (err) {
      setMediaStatus(`camera/mic permission denied: ${err.message}`);
      return;
    }

    const media = createMediaLink();
    startWebcamCapture(stream, media);
    startMicCapture(stream, media);
  }

  startMediaBridge();
})();
