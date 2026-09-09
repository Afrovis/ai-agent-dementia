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
})();
