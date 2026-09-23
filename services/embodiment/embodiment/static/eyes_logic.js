// Pure bedside eye and clock calculations. Coordinates are normalised camera coordinates.
(function (root) {
  "use strict";
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const expressions = new Set(["open", "sleeping", "sleepy", "listening", "speaking"]);

  function expression(serverExpression, speaking) {
    return speaking ? "speaking" : expressions.has(serverExpression) ? serverExpression : "open";
  }

  function status({ alert = false, cameraOff = false, microphoneOff = false,
    reconnecting = false, speaking = false, expression: current = "open" } = {}) {
    if (alert) return { label: "Needs attention", dot: 1, blink: false };
    if (cameraOff) return { label: "Camera off", dot: 1, blink: false };
    if (microphoneOff) return { label: "Microphone off", dot: 1, blink: false };
    if (reconnecting) return { label: "Reconnecting", dot: 1, blink: true };
    if (speaking || current === "speaking") return { label: "Speaking", dot: .8, blink: false };
    if (current === "listening") return { label: "Listening", dot: .8, blink: false };
    if (current === "sleepy") return { label: "Winding down", dot: .5, blink: false };
    if (current === "sleeping") return { label: "Resting · monitoring", dot: .35, blink: false };
    return { label: "", dot: 0, blink: false };
  }

  function gazeTransform(pos) {
    return { pos, x: pos * 220, y: Math.abs(pos) * 10,
      leftScale: 1 + .14 * pos, rightScale: 1 - .14 * pos };
  }

  function gazePoint(x, y) {
    return gazeTransform(clamp(((1 - clamp(x, 0, 1)) - .5) * 2, -1, 1));
  }

  function gazeTarget(previous, next, deadZone = .03) {
    return Math.abs(next - previous) < deadZone ? previous : next;
  }

  // Critically damped acceleration, with a displacement limit in screen widths/s.
  function springStep(state, target, dt, width = 1440) {
    const seconds = clamp(dt, 0, .05);
    const range = width * 220 / 1440;
    const cap = width * .15 / range;
    const omega = 5.5;
    let velocity = clamp((state.velocity || 0) +
      (omega * omega * (target - state.pos) - 2 * omega * (state.velocity || 0)) * seconds,
    -cap, cap);
    let pos = state.pos + velocity * seconds;
    if ((target - state.pos) * (target - pos) <= 0) { pos = target; velocity = 0; }
    return { pos, velocity };
  }

  function noneTarget(lastPos, elapsedSeconds) {
    if (elapsedSeconds <= 5) return { pos: lastPos, brightness: 1 };
    const t = clamp((elapsedSeconds - 5) / 3, 0, 1);
    const eased = t * t * (3 - 2 * t);
    return { pos: lastPos * (1 - eased), brightness: 1 - .25 * eased };
  }

  function rms(buffer, noiseFloor = .015, ceiling = .18) {
    if (!buffer || !buffer.length) return 0;
    let sum = 0;
    const bytes = buffer instanceof Uint8Array;
    for (const sample of buffer) {
      const value = bytes ? (sample - 128) / 128 : sample;
      sum += value * value;
    }
    return clamp((Math.sqrt(sum / buffer.length) - noiseFloor) / (ceiling - noiseFloor), 0, 1);
  }

  function smoothAmp(current, target, dt) {
    const tau = target > current ? .08 : .3;
    return clamp(current + (target - current) * (1 - Math.exp(-Math.max(0, dt) / tau)), 0, 1);
  }

  function minutes(value) {
    const match = /^(\d{2}):(\d{2})$/.exec(value || "");
    return match && +match[1] < 24 && +match[2] < 60 ? +match[1] * 60 + +match[2] : null;
  }

  function isNight(date, config = {}) {
    const start = minutes(config.night_start) ?? 1200;
    const end = minutes(config.night_end) ?? 420;
    const now = date.getHours() * 60 + date.getMinutes();
    return start <= end ? now >= start && now < end : now >= start || now < end;
  }

  function formatClock(date, clock24h = false) {
    const hour = date.getHours();
    const minute = String(date.getMinutes()).padStart(2, "0");
    return clock24h ? `${String(hour).padStart(2, "0")}:${minute}` :
      `${hour % 12 || 12}:${minute} ${hour < 12 ? "AM" : "PM"}`;
  }

  const api = { expression, status, gazePoint, gazeTransform, gazeTarget, springStep, noneTarget,
    rms, smoothAmp, isNight, formatClock };
  root.EyesLogic = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window === "undefined" ? globalThis : window);
