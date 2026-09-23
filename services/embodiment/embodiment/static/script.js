// Night Companion embodiment page client.
//
// Connects to the `/ws` endpoint, applies `show`/`say` messages to the DOM,
// and reconnects with exponential backoff on disconnect. No framework, per
// HANDOFF.md section 4 ("Embodiment page is plain HTML, CSS, and JS").

(function () {
  "use strict";

  const logic = window.EyesLogic;
  const faceEl = document.getElementById("face");
  const eyeMotionEl = document.getElementById("eye-motion");
  const eyePulseEl = document.getElementById("eye-pulse");
  const eyeLeftEl = document.querySelector(".eye-left");
  const eyeRightEl = document.querySelector(".eye-right");
  const vignetteEl = document.getElementById("alert-vignette");
  const statusDotEl = document.getElementById("status-dot");
  const statusLabelEl = document.getElementById("status-label");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const eyes = { serverExpression: "sleeping", currentExpression: "sleeping", alert: false,
    gaze: { target: "none", x: null, y: null }, targetPos: 0, lastPos: 0,
    noneAt: performance.now(), spring: { pos: 0, velocity: 0 }, amp: 0,
    cameraOff: true, microphoneOff: true, config: { night_start: "20:00", night_end: "07:00", clock_24h: false } };
  let micAnalyser = null;
  let speechAnalyser = null;
  let demoAmp = null;
  let blinkHoldTimer = null;
  let statusTimer = null;
  let alertTimer = null;

  function renderClock() {
    const now = new Date();
    document.getElementById("clock-time").textContent = logic.formatClock(now, eyes.config.clock_24h);
    document.getElementById("clock-icon").classList.toggle("day", !logic.isNight(now, eyes.config));
  }
  renderClock();
  setInterval(renderClock, 10000);

  function renderStatus() {
    const result = logic.status({ alert: eyes.alert, cameraOff: eyes.cameraOff,
      microphoneOff: eyes.microphoneOff, reconnecting: !debugState.ws,
      speaking: debugState.speaking, expression: eyes.currentExpression });
    if (statusLabelEl.textContent !== result.label) {
      statusLabelEl.style.opacity = "0";
      clearTimeout(statusTimer);
      statusTimer = setTimeout(() => {
        statusLabelEl.textContent = result.label;
        statusLabelEl.style.opacity = result.label ? "1" : "0";
      }, 300);
    }
    statusDotEl.style.opacity = String(result.dot);
    statusDotEl.classList.toggle("reconnecting", result.blink);
  }

  function renderExpression() {
    const next = logic.expression(eyes.serverExpression, debugState.speaking);
    if (next !== eyes.currentExpression) {
      eyes.currentExpression = next;
      faceEl.className = `face ${next} blink-hold`;
      faceEl.setAttribute("aria-label", `companion ${next}`);
      clearTimeout(blinkHoldTimer);
      blinkHoldTimer = setTimeout(() => faceEl.classList.remove("blink-hold"), 2000);
    }
    renderStatus();
  }

  function applyEyes(msg) {
    if (["open", "sleeping", "sleepy", "listening", "speaking"].includes(msg.expression)) eyes.serverExpression = msg.expression;
    if (typeof msg.alert === "boolean" && msg.alert !== eyes.alert) {
      eyes.alert = msg.alert;
      // Pin the glow where the pulse left it, with transitions off and a style
      // flush, so the 3 s fade starts from there instead of snapping to zero.
      const currentOpacity = getComputedStyle(vignetteEl).opacity;
      vignetteEl.style.transition = "none";
      vignetteEl.classList.remove("pulsing", "on");
      vignetteEl.style.opacity = currentOpacity;
      void getComputedStyle(vignetteEl).opacity;
      vignetteEl.style.transition = "";
      clearTimeout(alertTimer);
      requestAnimationFrame(() => {
        vignetteEl.style.opacity = eyes.alert ? (reducedMotion.matches ? ".30" : ".15") : "0";
        if (eyes.alert) alertTimer = setTimeout(() => {
          vignetteEl.style.opacity = "";
          vignetteEl.classList.add("pulsing");
        }, 3000);
      });
    }
    if (msg.gaze && ["face", "bed", "none"].includes(msg.gaze.target)) {
      const gaze = msg.gaze;
      if (gaze.target === "none") {
        if (eyes.gaze.target !== "none") { eyes.noneAt = performance.now(); eyes.lastPos = eyes.spring.pos; }
      } else if (Number.isFinite(gaze.x) && Number.isFinite(gaze.y)) {
        eyes.targetPos = logic.gazeTarget(eyes.targetPos, logic.gazePoint(gaze.x, gaze.y).pos);
      }
      eyes.gaze = gaze;
    }
    renderExpression();
  }

  function readLoudness(analyser) {
    if (!analyser || !micAudioContext || micAudioContext.state !== "running") return 0;
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    return logic.rms(data);
  }

  let lastFrameTime = performance.now();
  function animateEyes(now) {
    const dt = Math.min(.05, Math.max(0, (now - lastFrameTime) / 1000));
    lastFrameTime = now;
    const none = eyes.gaze.target === "none" ? logic.noneTarget(eyes.lastPos, (now - eyes.noneAt) / 1000) : null;
    const target = none ? none.pos : eyes.targetPos;
    eyes.spring = reducedMotion.matches ? { pos: 0, velocity: 0 } :
      logic.springStep(eyes.spring, target, dt, innerWidth);
    const point = logic.gazeTransform(eyes.spring.pos);
    const scale = Math.min(innerWidth / 1440, innerHeight / 900);
    eyeMotionEl.style.transform = `translate(${point.x * scale}px, ${point.y * scale}px)`;
    eyeMotionEl.style.opacity = String(none ? none.brightness : 1);
    eyeLeftEl.style.transform = `scale(${point.leftScale})`;
    eyeRightEl.style.transform = `scale(${point.rightScale})`;
    const active = eyes.currentExpression === "listening" || eyes.currentExpression === "speaking";
    const analyser = eyes.currentExpression === "listening" ? micAnalyser : speechAnalyser;
    const raw = demoAmp === null ? active ? readLoudness(analyser) : 0 : demoAmp;
    eyes.amp = reducedMotion.matches ? 0 : logic.smoothAmp(eyes.amp, raw, dt);
    const glow = eyes.currentExpression === "sleeping" ? 6 : eyes.currentExpression === "listening" ?
      26 + 22 * eyes.amp : eyes.currentExpression === "speaking" ? 14 + 16 * eyes.amp : 14;
    eyePulseEl.style.filter = `drop-shadow(0 0 ${glow * scale}px #B8322A)`;
    eyePulseEl.style.transform = active && !reducedMotion.matches ? `scale(${1 + .03 * eyes.amp})` : "scale(1)";
    requestAnimationFrame(animateEyes);
  }
  requestAnimationFrame(animateEyes);
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
    pendingControl: null, pendingUntil: 0,
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
      return true;
    }
    return false;
  }

  // Clicks step from the last requested control, not the agent's last echo, so
  // quick repeated clicks are not lost while the echo is still on its way.
  function requestedControl() {
    if (debugState.pendingControl && Date.now() >= debugState.pendingUntil) debugState.pendingControl = null;
    return debugState.pendingControl || debugState.appliedControl;
  }
  function requestControl(control) {
    if (sendDebug({ type: "debug_control", ...control })) {
      debugState.pendingControl = control;
      debugState.pendingUntil = Date.now() + 3000;
      renderControls();
    }
  }
  controlsEl.querySelectorAll("[data-offset]").forEach((button) => {
    button.addEventListener("click", () => {
      const current = requestedControl();
      const step = Number(button.dataset.offset);
      requestControl({
        time_offset_hours: step === 0 ? 0 : Math.max(-23, Math.min(23, current.time_offset_hours + step)),
        force_in_bed: current.force_in_bed });
    });
  });
  document.getElementById("force-in-bed").addEventListener("click", () => {
    const current = requestedControl();
    requestControl({ time_offset_hours: current.time_offset_hours, force_in_bed: !current.force_in_bed });
  });
  const resetSession = document.getElementById("reset-session");
  let resetDeadline = 0;
  let resetTimer = null;
  resetSession.addEventListener("click", () => {
    if (Date.now() < resetDeadline) {
      resetDeadline = 0;
      clearTimeout(resetTimer);
      if (sendDebug({ type: "reset_session" })) {
        resetSession.textContent = "Session reset sent";
        resetTimer = setTimeout(() => { resetSession.textContent = "Reset session"; }, 3000);
      } else {
        resetSession.textContent = "Reset session";
      }
      return;
    }
    clearTimeout(resetTimer);
    resetDeadline = Date.now() + 3000;
    resetSession.textContent = "Confirm reset?";
    resetTimer = setTimeout(() => {
      resetDeadline = 0;
      resetSession.textContent = "Reset session";
    }, 3000);
  });
  document.getElementById("detect-bed").addEventListener("click", () => sendDebug({ type: "calibrate_bed" }));

  function renderControls() {
    controlsEl.classList.toggle("hidden", !debugState.controlsEnabled);
    const requested = requestedControl();
    const waiting = requested !== debugState.appliedControl ? "…" : "";
    const requestedOffset = requested.time_offset_hours;
    const suffix = requestedOffset === 0 ? "" : ` (${requestedOffset > 0 ? "+" : "−"}${Math.abs(requestedOffset)}h)`;
    const agentTime = new Date(Date.now() + requestedOffset * 3600000);
    document.getElementById("agent-clock").textContent = `Agent clock ${agentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}${suffix}${waiting}`;
    document.getElementById("force-in-bed").textContent = `Force in bed: ${requested.force_in_bed ? "ON" : "OFF"}${waiting}`;
    // The override banner shows what the agent has actually applied.
    const applied = debugState.appliedControl;
    const offset = applied.time_offset_hours;
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

  let currentSpeech = null;
  let currentSpeechInterruptible = false;
  let currentSpeechSessionId = null;
  let currentSpeechRecord = null;
  let pendingSpeech = null;
  let playbackSocket = null;
  let micAudioContext = null;

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
    speechAnalyser = null;
    renderExpression();
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
      renderExpression();
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
    if (!micAudioContext) return;
    micAudioContext.resume().then(() => {
      if (micAudioContext.state !== "running") return;
      const source = micAudioContext.createBufferSource();
      source.buffer = micAudioContext.createBuffer(1, 1, micAudioContext.sampleRate);
      source.connect(micAudioContext.destination);
      source.start();
      if (debugState.audioBlocked || !debugState.audioUnlocked) {
        debugState.audioUnlocked = true;
        debugState.audioBlocked = false;
        sendIdentity("heartbeat");
        audioUnlockEl.classList.add("hidden");
        reportPlayback(record || { strategy: null, sessionId: null, audioId: null, receivedAt: performance.now() }, "unlocked");
      }
    }).catch(() => { /* the next gesture can retry */ });
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
    // A media element may be attached to a source once. Each Say gets a new Audio.
    if (micAudioContext && micAudioContext.state === "running") {
      try {
        const source = micAudioContext.createMediaElementSource(speech);
        const analyser = micAudioContext.createAnalyser();
        analyser.fftSize = 1024;
        source.connect(analyser);
        analyser.connect(micAudioContext.destination);
        speechAnalyser = analyser;
      } catch (error) { console.error("speech analyser unavailable", error); }
    } else speechAnalyser = null;
    record.speech = speech;
    currentSpeech = speech;
    currentSpeechRecord = record;
    currentSpeechInterruptible = msg.interruptible === true;
    currentSpeechSessionId = msg.session_id || null;
    speech.addEventListener("play", () => {
      if (currentSpeechRecord === record) { debugState.speaking = true; renderExpression(); }
    });
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
      renderExpression();
      if (wasBlocked) reportPlayback(record, "unlocked");
      reportPlayback(record, "playing");
    });
    speech.addEventListener("pause", () => {
      if (currentSpeechRecord === record) { debugState.speaking = false; renderExpression(); }
    });
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

  }

  function handleMessage(event) {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error("could not parse websocket message", err);
      return;
    }
    if (msg.type === "eyes") {
      applyEyes(msg);
    } else if (msg.type === "config") {
      eyes.config = { ...eyes.config, ...msg };
      renderClock();
    } else if (msg.type === "show") {
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
      const pending = debugState.pendingControl;
      if (pending && pending.time_offset_hours === msg.time_offset_hours && pending.force_in_bed === msg.force_in_bed) {
        debugState.pendingControl = null;
      }
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
      renderStatus();
      identitySockets.add(socket);
      socket.send(JSON.stringify(identityMessage("hello")));
      console.log("embodiment websocket connected");
    });

    socket.addEventListener("message", handleMessage);

    socket.addEventListener("close", () => {
      identitySockets.delete(socket);
      if (playbackSocket === socket) playbackSocket = null;
      debugState.ws = false;
      renderStatus();
      const nextBackoff = Math.min(currentBackoff * 2, 30000);
      setTimeout(() => connect(nextBackoff), currentBackoff);
    });

    socket.addEventListener("error", () => {
      socket.close();
    });
  }

  connect(1000);
  renderStatus();
  if (new URLSearchParams(location.search).get("demo") === "1") {
    window.__eyesDemo = {
      receive: (msg) => handleMessage({ data: JSON.stringify(msg) }),
      setAmp: (value) => { demoAmp = value === null ? null : Math.min(1, Math.max(0, Number(value) || 0)); },
    };
  }

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
    audioContext.resume().catch(() => { /* first pointer gesture retries */ });
    const source = audioContext.createMediaStreamSource(stream);
    micAnalyser = audioContext.createAnalyser();
    micAnalyser.fftSize = 1024;
    source.connect(micAnalyser);
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
      renderStatus();
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
      renderStatus();
      return;
    }

    eyes.cameraOff = stream.getVideoTracks().length === 0;
    eyes.microphoneOff = stream.getAudioTracks().length === 0;
    for (const track of stream.getVideoTracks()) track.addEventListener("ended", () => {
      eyes.cameraOff = true; renderStatus();
    });
    for (const track of stream.getAudioTracks()) track.addEventListener("ended", () => {
      eyes.microphoneOff = true; renderStatus();
    });
    renderStatus();
    const media = createMediaLink();
    startWebcamCapture(stream, media);
    startMicCapture(stream, media);
  }

  startMediaBridge();
})();
