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
  const audioUnlockEl = document.getElementById("audio-unlock");
  const debugEl = document.getElementById("debug");
  const debugCanvas = document.getElementById("debug-camera");
  const debugCtx = debugCanvas.getContext("2d");
  const pageLoadTime = new Date().toISOString();
  const randomId = () => crypto.randomUUID ? crypto.randomUUID() :
    Array.from(crypto.getRandomValues(new Uint8Array(16)), (value) => value.toString(16).padStart(2, "0")).join("");
  const pageId = randomId();
  let deviceId;
  try {
    deviceId = localStorage.getItem("night-companion-device-id");
    if (!deviceId) {
      deviceId = randomId();
      localStorage.setItem("night-companion-device-id", deviceId);
    }
  } catch (_) { deviceId = randomId(); }
  const browser = /Edg\//.test(navigator.userAgent) ? "Edge" :
    /Firefox\//.test(navigator.userAgent) ? "Firefox" :
    /Chrome\//.test(navigator.userAgent) ? "Chrome" :
    /Safari\//.test(navigator.userAgent) ? "Safari" : "unknown";
  const identitySockets = new Set();
  const debugState = {
    person: null, session: null, pose: null, video: null, ws: false, media: false,
    frameTimes: [], lastFrame: 0, lastAudio: 0, micLevel: 0,
    hearingUntil: 0, active: {}, speaking: false, audioUnlocked: false,
    audioBlocked: false, events: [], personChanged: Date.now(), pages: [],
    controlsEnabled: false, appliedControl: { time_offset_hours: 0, force_in_bed: false },
    bedZone: null,
  };
  const controlsEl = document.getElementById("debug-controls");
  // Control gestures must not unlock or retry patient-facing speech.
  for (const kind of ["pointerdown", "click"]) {
    controlsEl.addEventListener(kind, (event) => event.stopPropagation());
  }
  controlsEl.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() !== "d") event.stopPropagation();
  });

  function sendDebug(message) {
    if (debugState.controlsEnabled && playbackSocket && playbackSocket.readyState === WebSocket.OPEN) {
      playbackSocket.send(JSON.stringify(message));
    }
  }

  controlsEl.querySelectorAll("[data-offset]").forEach((button) => {
    button.addEventListener("click", () => {
      const applied = debugState.appliedControl;
      const step = Number(button.dataset.offset);
      sendDebug({ type: "debug_control",
        time_offset_hours: step === 0 ? 0 : Math.max(-23, Math.min(23, applied.time_offset_hours + step)),
        force_in_bed: applied.force_in_bed });
    });
  });
  document.getElementById("force-in-bed").addEventListener("click", () => {
    const applied = debugState.appliedControl;
    sendDebug({ type: "debug_control", time_offset_hours: applied.time_offset_hours,
      force_in_bed: !applied.force_in_bed });
  });
  document.getElementById("detect-bed").addEventListener("click", () => sendDebug({ type: "calibrate_bed" }));

  function renderControls() {
    controlsEl.classList.toggle("hidden", !debugState.controlsEnabled);
    const applied = debugState.appliedControl;
    const offset = applied.time_offset_hours;
    const suffix = offset === 0 ? "" : ` (${offset > 0 ? "+" : "−"}${Math.abs(offset)}h)`;
    const agentTime = new Date(Date.now() + offset * 3600000);
    document.getElementById("agent-clock").textContent = `Agent clock ${agentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}${suffix}`;
    document.getElementById("force-in-bed").textContent = `Force in bed: ${applied.force_in_bed ? "ON" : "OFF"}`;
    const overrides = [offset !== 0 ? `${offset > 0 ? "+" : "−"}${Math.abs(offset)}h` : "", applied.force_in_bed ? "in bed forced" : ""].filter(Boolean).join(" · ");
    const active = !!overrides;
    const banner = document.getElementById("debug-override-banner");
    const badge = document.getElementById("debug-override-badge");
    banner.textContent = `DEBUG OVERRIDE ACTIVE: ${overrides}`;
    badge.textContent = `DEBUG ${overrides}`;
    banner.classList.toggle("hidden", !active);
    badge.classList.toggle("hidden", !active);
    const bed = debugState.bedZone;
    const label = document.getElementById("bed-zone-label");
    const detail = document.getElementById("bed-zone-detail");
    const detect = document.getElementById("detect-bed");
    label.textContent = !bed ? "Bed zone: waiting for status" : bed.calibration === "running" ? "Detecting bed zone…" : bed.has_bed ? "Bed zone set" : "No bed zone set";
    detail.textContent = !bed ? "" : bed.calibration === "failed" ? (bed.detail || "Bed detection failed") : (!bed.has_bed && bed.detail) || "";
    detect.classList.toggle("hidden", !bed || (bed.has_bed && bed.calibration !== "running"));
    if (bed && !bed.has_bed) detect.classList.remove("hidden");
    detect.disabled = !!bed && bed.calibration === "running";
    detect.textContent = detect.disabled ? "Detecting bed zone…" : "Detect bed zone";
  }
  setInterval(renderControls, 1000);
  renderControls();
  function identityMessage(type) {
    return { type, page_id: pageId, device_id: deviceId,
      user_agent: navigator.userAgent.slice(0, 256), platform: (navigator.platform || "unknown").slice(0, 80),
      screen: { width: screen.width, height: screen.height },
      visibility: document.visibilityState,
      audio_unlocked: debugState.audioUnlocked && !debugState.audioBlocked,
      page_load_time: pageLoadTime };
  }
  function sendIdentity(type) {
    const payload = JSON.stringify(identityMessage(type));
    for (const socket of identitySockets) {
      if (socket.readyState === WebSocket.OPEN) socket.send(payload);
    }
  }
  document.addEventListener("visibilitychange", () => sendIdentity("visibility"));
  setInterval(() => sendIdentity("heartbeat"), 10000);
  let debugVisible = false;
  const skeleton = [
    ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"], ["left_hip", "right_hip"],
    ["left_hip", "left_knee"], ["right_hip", "right_knee"],
    ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
  ];

  function addDebugEvent(text, ts) {
    const when = ts ? new Date(ts) : new Date();
    debugState.events.unshift(`${when.toLocaleTimeString("en-GB", { hour12: false })} ${text}`);
    debugState.events.length = Math.min(debugState.events.length, 15);
  }

  function setDebugVisible(visible) {
    debugVisible = visible;
    debugEl.classList.toggle("hidden", !visible);
    try { localStorage.setItem("night-companion-debug", visible ? "1" : "0"); } catch (_) { /* storage may be disabled */ }
    if (visible) requestAnimationFrame(drawDebugCamera);
  }

  let storedDebug = false;
  try { storedDebug = localStorage.getItem("night-companion-debug") === "1"; } catch (_) { /* storage may be disabled */ }
  setDebugVisible(new URLSearchParams(location.search).get("debug") === "1" || storedDebug);
  document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "d" && !event.altKey && !event.ctrlKey && !event.metaKey && !event.repeat) {
      setDebugVisible(!debugVisible);
    }
  });

  function drawDebugCamera() {
    if (!debugVisible) return;
    const ctx = debugCtx;
    const width = debugCanvas.width;
    const height = debugCanvas.height;
    ctx.fillStyle = "#171c22";
    ctx.fillRect(0, 0, width, height);
    const video = debugState.video;
    if (video && video.readyState >= video.HAVE_CURRENT_DATA) {
      // Pose coordinates refer to the 640x480 transport frame, including its letterbox.
      const transport = letterboxRect(FRAME_WIDTH, FRAME_HEIGHT, width, height);
      ctx.fillStyle = "black";
      ctx.fillRect(transport.x, transport.y, transport.width, transport.height);
      const image = letterboxRect(video.videoWidth, video.videoHeight, transport.width, transport.height);
      ctx.drawImage(video, transport.x + image.x, transport.y + image.y, image.width, image.height);
      const polygon = debugState.bedZone && debugState.bedZone.polygon;
      if (polygon && polygon.length > 2) {
        ctx.beginPath();
        polygon.forEach(([px, py], index) => {
          const x = transport.x + px * transport.width;
          const y = transport.y + py * transport.height;
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffbd69";
        ctx.stroke();
      }
      const pose = debugState.pose;
      if (pose && pose.detected) {
        const age = (Date.now() - new Date(pose.ts).getTime()) / 1000;
        const fade = age > 3 ? 0.25 : 1;
        const x = (value) => transport.x + value * transport.width;
        const y = (value) => transport.y + value * transport.height;
        ctx.lineWidth = 2;
        ctx.strokeStyle = `rgba(56, 230, 145, ${fade})`;
        if (pose.bbox) {
          ctx.strokeRect(x(pose.bbox[0]), y(pose.bbox[1]), (pose.bbox[2] - pose.bbox[0]) * transport.width, (pose.bbox[3] - pose.bbox[1]) * transport.height);
        }
        const points = pose.landmarks || {};
        const lines = skeleton.slice();
        if (points.nose && points.left_shoulder && points.right_shoulder) {
          const mid = [(points.left_shoulder[0] + points.right_shoulder[0]) / 2, (points.left_shoulder[1] + points.right_shoulder[1]) / 2];
          ctx.beginPath(); ctx.moveTo(x(points.nose[0]), y(points.nose[1])); ctx.lineTo(x(mid[0]), y(mid[1])); ctx.stroke();
        }
        lines.forEach(([a, b]) => {
          if (!points[a] || !points[b]) return;
          ctx.globalAlpha = Math.min(points[a][2], points[b][2]) * fade;
          ctx.beginPath(); ctx.moveTo(x(points[a][0]), y(points[a][1])); ctx.lineTo(x(points[b][0]), y(points[b][1])); ctx.stroke();
        });
        Object.values(points).forEach((point) => {
          ctx.globalAlpha = point[2] * fade;
          ctx.fillStyle = "#ffdb70";
          ctx.beginPath(); ctx.arc(x(point[0]), y(point[1]), 3, 0, Math.PI * 2); ctx.fill();
        });
        ctx.globalAlpha = 1;
      }
    } else {
      ctx.fillStyle = "#aab5c0";
      ctx.fillText("camera off", 12, 20);
    }
    requestAnimationFrame(drawDebugCamera);
  }

  function indicator(label, active, extra = "", rec = false) {
    return `<span class="debug-indicator${active ? " active" : ""}${rec ? " rec" : ""}">${label}${extra}</span>`;
  }

  function renderDebugStatus() {
    if (!debugVisible) return;
    const now = Date.now();
    const person = debugState.person;
    const pose = debugState.pose;
    const state = person ? person.state : "unknown";
    const chip = document.getElementById("debug-person");
    chip.className = `state-chip ${state}`;
    chip.textContent = state;
    document.getElementById("debug-person-detail").textContent = person ? `${Math.round(person.confidence * 100)}% · ${person.zone} · ${((now - debugState.personChanged) / 1000).toFixed(0)}s` : "";
    const session = debugState.session;
    document.getElementById("debug-session").textContent = session ? `Session: ${session.phase} · ${session.goal} · strategy ${session.strategy_index}` : "Session: unknown";
    document.getElementById("debug-client").textContent = `Page ${pageId.slice(0, 8)} · ${browser} · ${document.visibilityState} · audio:${debugState.audioUnlocked && !debugState.audioBlocked ? "ok" : "blocked"}`;
    document.getElementById("debug-clients").textContent = `${debugState.pages.length} page${debugState.pages.length === 1 ? "" : "s"} connected`;
    document.getElementById("debug-pose-label").textContent = !debugState.video ? "camera off" : pose ? `pose ${Math.round(pose.confidence * 100)}% · ${pose.candidate_state || "—"} · age ${((now - new Date(pose.ts).getTime()) / 1000).toFixed(1)}s · ${Math.round(pose.latency_ms)}ms` : "waiting for pose";
    debugState.frameTimes = debugState.frameTimes.filter((t) => now - t < 5000);
    const cam = debugState.media && now - debugState.lastFrame < 2500;
    const mic = debugState.media && now - debugState.lastAudio < 2000;
    const active = debugState.active;
    document.getElementById("debug-indicators").innerHTML = [
      indicator("CAM", cam, ` ${Math.round(debugState.frameTimes.length / 5)} fps`),
      indicator("MIC", mic, ` <span class="mic-level"><i style="width:${Math.round(debugState.micLevel * 100)}%"></i></span>`),
      indicator("WS", debugState.ws), indicator("MEDIA", debugState.media),
      indicator("HEARING", debugState.hearingUntil > now),
      indicator("STT", !!active.transcribe),
      indicator("THINK", !!active.interpret || !!active.compose, active.compose ? " compose" : active.interpret ? " interpret" : ""),
      indicator("TTS", !!active.tts), indicator("SPEAK", debugState.speaking),
      `<span class="debug-indicator${debugState.audioUnlocked ? " active" : ""}${debugState.audioBlocked ? " blocked" : ""}">AUDIO${debugState.audioBlocked ? " blocked" : ""}</span>`,
      indicator("REC", cam || mic, "", true),
    ].join("");
    const log = document.getElementById("debug-events");
    log.replaceChildren(...debugState.events.map((entry) => { const row = document.createElement("div"); row.className = "debug-event"; row.textContent = entry; return row; }));
    document.getElementById("debug-clock").textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
  }
  setInterval(renderDebugStatus, 500);

  const FACE_STATES = ["asleep", "awake", "speaking", "listening"];
  let currentSpeech = null;
  let currentSpeechInterruptible = false;
  let currentSpeechSessionId = null;
  let currentSpeechRecord = null;
  let pendingSpeech = null;
  let playbackSocket = null;
  let micAudioContext = null;
  let unlockContext = null;

  function reportPlayback(record, phase, detail = "", error = null) {
    const payload = {
      type: "playback", phase,
      strategy: record.strategy,
      session_id: record.sessionId,
      audio_id: record.audioId,
      latency_ms: Math.round(performance.now() - record.receivedAt),
      detail,
      error_name: error ? String(error.name || "Error").slice(0, 64) : null,
      error_message: error ? String(error.message || "").slice(0, 160) : null,
    };
    const label = `say ${record.strategy || "unknown"}`;
    if (phase === "failed") addDebugEvent(`${label}: ${payload.error_name === "NotAllowedError" ? "blocked" : "failed"} ${payload.error_name}${payload.error_message ? `: ${payload.error_message}` : ""}`);
    else if (phase === "ended") addDebugEvent(`${label}: played ${(record.playedMs / 1000).toFixed(1)} s`);
    else if (phase === "interrupted") addDebugEvent(`${label}: interrupted ${detail}`);
    else if (phase === "no_audio") addDebugEvent(`${label}: no audio`);
    else addDebugEvent(`${label}: ${phase}`);
    if (playbackSocket && playbackSocket.readyState === WebSocket.OPEN) {
      playbackSocket.send(JSON.stringify(payload));
    }
  }

  function audioId(url) {
    const match = /\/(?:speech|voice)\/([^/?#]+)\.wav(?:[?#]|$)/.exec(url);
    return match ? match[1].slice(0, 24) : null;
  }

  function clearSpeech(record) {
    if (currentSpeechRecord !== record) return;
    if (pendingSpeech === record) pendingSpeech = null;
    audioUnlockEl.classList.add("hidden");
    currentSpeech = null;
    currentSpeechRecord = null;
    currentSpeechInterruptible = false;
    currentSpeechSessionId = null;
    debugState.speaking = false;
    if (faceEl.classList.contains("speaking")) {
      faceEl.classList.replace("speaking", "awake");
    }
  }

  function interruptSpeech(why) {
    if (!currentSpeech || !currentSpeechRecord) return;
    const record = currentSpeechRecord;
    currentSpeech.pause();
    currentSpeech.currentTime = 0;
    if (pendingSpeech === record) pendingSpeech = null;
    reportPlayback(record, "interrupted", why);
    clearSpeech(record);
  }

  function requestSpeech(record) {
    if (currentSpeechRecord !== record) return;
    reportPlayback(record, "requested");
    record.speech.play().catch((error) => {
      if (currentSpeechRecord !== record) return;
      debugState.speaking = false;
      reportPlayback(record, "failed", "", error);
      if (error.name === "NotAllowedError") {
        debugState.audioBlocked = true;
        debugState.audioUnlocked = false;
        sendIdentity("heartbeat");
        pendingSpeech = record;
        audioUnlockEl.classList.remove("hidden");
      } else {
        clearSpeech(record);
      }
      console.error("Piper speech playback failed", error);
    });
  }

  function unlockAudioFromGesture(event) {
    if (event && controlsEl.contains(event.target)) return;
    // Call play() synchronously inside the gesture; browser activation can be
    // lost after the first await. A blocked clip is retried only while fresh.
    const record = pendingSpeech;
    audioUnlockEl.classList.add("hidden");
    if (record && currentSpeechRecord === record && Date.now() - record.receivedWall < 20000) {
      pendingSpeech = null;
      requestSpeech(record);
    } else {
      pendingSpeech = null;
    }
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    try {
      unlockContext = unlockContext || new AudioContextClass();
      for (const context of [unlockContext, micAudioContext]) {
        if (!context) continue;
        context.resume().then(() => {
          if (context === unlockContext && context.state === "running") {
            const source = context.createBufferSource();
            source.buffer = context.createBuffer(1, 1, context.sampleRate);
            source.connect(context.destination);
            source.start();
            if (debugState.audioBlocked || !debugState.audioUnlocked) {
              debugState.audioUnlocked = true;
              debugState.audioBlocked = false;
              sendIdentity("heartbeat");
              audioUnlockEl.classList.add("hidden");
              reportPlayback(record || { strategy: null, sessionId: null, audioId: null, receivedAt: performance.now() }, "unlocked");
            }
          }
        }).catch(() => { /* the next gesture can retry */ });
      }
    } catch (error) {
      console.error("audio unlock failed", error);
    }
  }

  document.addEventListener("pointerdown", unlockAudioFromGesture);
  document.addEventListener("keydown", unlockAudioFromGesture);
  document.getElementById("audio-unlock-button").addEventListener("click", unlockAudioFromGesture);

  // A photo the caregiver never uploaded (or has since deleted) answers 404.
  // Fade the layer back out and hide it rather than leaving the browser's
  // broken-image glyph on a bedroom wall at 3am.
  photoImgEl.addEventListener("error", () => {
    photoEl.classList.remove("visible");
    photoEl.classList.add("hidden");
  });

  function applyShow(msg) {
    addDebugEvent(`show: ${msg.face}`);
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
    interruptSpeech("replaced by newer Say");
    if (typeof msg.text === "string") {
      bodyEl.textContent = msg.text;
    }
    const record = {
      strategy: typeof msg.strategy === "string" ? msg.strategy.slice(0, 128) : null,
      sessionId: typeof msg.session_id === "string" ? msg.session_id.slice(0, 128) : null,
      audioId: typeof msg.audio_url === "string" ? audioId(msg.audio_url) : null,
      receivedAt: performance.now(), receivedWall: Date.now(), playedMs: 0, playingAt: null,
    };
    reportPlayback(record, "received");
    if (typeof msg.audio_url !== "string" || !msg.audio_url) {
      reportPlayback(record, "no_audio");
      return;
    }
    const speech = new Audio(msg.audio_url);
    record.speech = speech;
    currentSpeech = speech;
    currentSpeechRecord = record;
    currentSpeechInterruptible = msg.interruptible === true;
    currentSpeechSessionId = msg.session_id || null;
    speech.addEventListener("playing", () => {
      if (currentSpeechRecord !== record) return;
      const wasBlocked = debugState.audioBlocked;
      debugState.speaking = true;
      debugState.audioUnlocked = true;
      debugState.audioBlocked = false;
      sendIdentity("heartbeat");
      pendingSpeech = null;
      audioUnlockEl.classList.add("hidden");
      record.playingAt = performance.now();
      faceEl.classList.remove(...FACE_STATES);
      faceEl.classList.add("speaking");
      if (wasBlocked) reportPlayback(record, "unlocked");
      reportPlayback(record, "playing");
    });
    speech.addEventListener("pause", () => { if (currentSpeechRecord === record) debugState.speaking = false; });
    speech.addEventListener("ended", () => {
      if (currentSpeechRecord !== record) return;
      if (record.playingAt !== null) record.playedMs += performance.now() - record.playingAt;
      reportPlayback(record, "ended");
      clearSpeech(record);
    });
    speech.addEventListener("error", () => {
      if (currentSpeechRecord !== record) return;
      const names = { 1: "AbortError", 2: "NetworkError", 3: "EncodingError", 4: "NotSupportedError" };
      const error = { name: names[speech.error?.code] || "MediaError", message: speech.error?.message || "audio load failed" };
      reportPlayback(record, "failed", "", error);
      clearSpeech(record);
    });
    requestSpeech(record);
  }

  function applySpeechStarted(msg) {
    debugState.hearingUntil = Date.now() + 15000;
    if (!currentSpeech || !currentSpeechInterruptible) {
      return;
    }
    if (msg.session_id && currentSpeechSessionId !== msg.session_id) {
      return;
    }
    interruptSpeech("barge-in");
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
    } else if (msg.type === "person") {
      if (!debugState.person || debugState.person.state !== msg.state) {
        debugState.personChanged = Date.parse(msg.ts) || Date.now();
        addDebugEvent(`person: ${msg.state}`, msg.ts);
      }
      debugState.person = msg;
    } else if (msg.type === "session") {
      if (!debugState.session || debugState.session.phase !== msg.phase) addDebugEvent(`phase: ${msg.phase}`, msg.ts);
      debugState.session = msg;
    } else if (msg.type === "utterance") {
      debugState.hearingUntil = 0;
      addDebugEvent(`heard: ${msg.text}`, msg.ts);
    } else if (msg.type === "pose") {
      debugState.pose = msg;
    } else if (msg.type === "debug_config") {
      debugState.controlsEnabled = msg.enabled === true;
      renderControls();
    } else if (msg.type === "debug_state") {
      debugState.appliedControl = msg;
      renderControls();
    } else if (msg.type === "bed_zone") {
      debugState.bedZone = msg;
      renderControls();
    } else if (msg.type === "activity") {
      if (msg.kind === "playback") return;
      debugState.active[msg.kind] = msg.phase === "start";
      if (msg.phase === "end") addDebugEvent(`${msg.kind} ${Math.round(msg.duration_ms || 0)} ms ${msg.ok ? "ok" : msg.detail || "failed"}`, msg.ts);
    } else if (msg.type === "clients") {
      debugState.pages = msg.pages;
      const warning = document.getElementById("clients-warning");
      warning.textContent = `${msg.pages.length} pages connected — speech plays on all of them`;
      warning.classList.toggle("hidden", msg.pages.length <= 1);
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
    playbackSocket = socket;

    socket.addEventListener("open", () => {
      debugState.ws = true;
      identitySockets.add(socket);
      socket.send(JSON.stringify(identityMessage("hello")));
      console.log("embodiment websocket connected");
    });

    socket.addEventListener("message", handleMessage);

    socket.addEventListener("close", () => {
      identitySockets.delete(socket);
      if (playbackSocket === socket) playbackSocket = null;
      debugState.ws = false;
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
  const FRAME_WIDTH = 640;
  const FRAME_HEIGHT = 480;
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
  // changing its proportions. A 16:9 stream becomes 640x360 at (0, 60).
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
        identitySockets.add(socket);
        socket.send(JSON.stringify(identityMessage("hello")));
        debugState.media = true;
        setMediaStatus("media bridge connected");
      });

      socket.addEventListener("close", () => {
        identitySockets.delete(socket);
        link.ready = false;
        debugState.media = false;
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
        return true;
      }
      return false;
    };
    return link;
  }

  function startWebcamCapture(stream, media) {
    const video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.play().catch((err) => console.error("webcam preview failed to start", err));
    debugState.video = video;

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
            if (media.send({
              type: "frame",
              jpeg_b64: jpegB64,
              width: FRAME_WIDTH,
              height: FRAME_HEIGHT,
              source_width: sourceWidth,
              source_height: sourceHeight,
            })) {
              debugState.lastFrame = Date.now();
              debugState.frameTimes.push(debugState.lastFrame);
            }
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
    micAudioContext = audioContext;
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
      let sumSquares = 0;
      for (let i = 0; i < pcm16.length; i++) sumSquares += (pcm16[i] / 32768) ** 2;
      debugState.micLevel = Math.min(1, Math.sqrt(sumSquares / Math.max(1, pcm16.length)) * 8);
      if (media.send({
        type: "audio",
        pcm16_b64: arrayBufferToBase64(pcm16.buffer),
        sample_rate: AUDIO_SAMPLE_RATE,
      })) debugState.lastAudio = Date.now();
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
