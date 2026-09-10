// Night Companion dashboard: Zones editor client.
//
// Plain HTML, CSS and JS, no framework, per HANDOFF.md section 4 -- the one
// exception is the vendored htmx.min.js used only for the save form's
// submission (PLAN.md section 9: "Dashboard uses HTMX"). Draws the three
// caregiver zones (bed, door, bathroom path) as polygons on a canvas over
// the most recent frame the dashboard has in memory, and keeps a hidden
// form field in sync with the in-progress drawing so the HTMX POST always
// carries the latest state.

(function () {
  "use strict";

  const ZONE_NAMES = ["bed", "door", "bathroom_path"];
  const ZONE_FILL = {
    bed: "rgba(232, 165, 92, 0.35)",
    door: "rgba(92, 168, 232, 0.35)",
    bathroom_path: "rgba(140, 232, 92, 0.35)",
  };
  const ZONE_STROKE = {
    bed: "#e8a55c",
    door: "#5ca8e8",
    bathroom_path: "#8ce85c",
  };
  const ZONE_LABELS = {
    bed: "Bed",
    door: "Door",
    bathroom_path: "Bathroom path",
  };
  const POINT_RADIUS = 5;
  const COORDINATE_DECIMALS = 4;

  const frameMissingEl = document.getElementById("frame-missing");
  const editorEl = document.getElementById("editor");
  const canvas = document.getElementById("zones-canvas");
  const ctx = canvas.getContext("2d");
  const hiddenInput = document.getElementById("zones-json-input");
  const zoneHintEl = document.getElementById("zone-hint");
  const saveResultEl = document.getElementById("save-result");

  // Points are stored normalised (0.0-1.0), never in pixels, so the saved
  // config survives a change of camera resolution (see zones_store.py).
  const zones = { bed: [], door: [], bathroom_path: [] };
  // A zone starts "closed" once loaded from the saved config (or once the
  // caregiver explicitly closes it), so an accidental canvas click does
  // not silently extend an already-drawn polygon.
  const zonesClosed = { bed: false, door: false, bathroom_path: false };
  let activeZone = "bed";

  const image = new Image();
  let imageLoaded = false;

  function round(value) {
    return Math.round(value * 10 ** COORDINATE_DECIMALS) / 10 ** COORDINATE_DECIMALS;
  }

  function updateHiddenInput() {
    hiddenInput.value = JSON.stringify(zones);
  }

  function drawPolygon(name) {
    const points = zones[name];
    if (points.length === 0) {
      return;
    }
    ctx.beginPath();
    ctx.moveTo(points[0][0] * canvas.width, points[0][1] * canvas.height);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0] * canvas.width, points[i][1] * canvas.height);
    }
    if (points.length > 2) {
      ctx.closePath();
      ctx.fillStyle = ZONE_FILL[name];
      ctx.fill();
    }
    ctx.strokeStyle = ZONE_STROKE[name];
    ctx.lineWidth = 2;
    ctx.stroke();

    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x * canvas.width, y * canvas.height, POINT_RADIUS, 0, 2 * Math.PI);
      ctx.fillStyle = ZONE_STROKE[name];
      ctx.fill();
    }

    if (points.length > 0) {
      const [labelX, labelY] = points[0];
      ctx.fillStyle = ZONE_STROKE[name];
      ctx.font = "16px sans-serif";
      ctx.fillText(ZONE_LABELS[name], labelX * canvas.width + 8, labelY * canvas.height - 8);
    }
  }

  function draw() {
    if (!imageLoaded) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const name of ZONE_NAMES) {
      drawPolygon(name);
    }
  }

  function updateZoneButtons() {
    document.querySelectorAll(".zone-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.zone === activeZone);
    });
    const closed = zonesClosed[activeZone];
    zoneHintEl.textContent = closed
      ? `${ZONE_LABELS[activeZone]} is closed. Click "Edit shape" to add more points.`
      : `Click on the image to add points to ${ZONE_LABELS[activeZone]}.`;
  }

  function setActiveZone(name) {
    activeZone = name;
    updateZoneButtons();
  }

  function canvasPointFromEvent(evt) {
    const rect = canvas.getBoundingClientRect();
    const x = (evt.clientX - rect.left) / rect.width;
    const y = (evt.clientY - rect.top) / rect.height;
    return [Math.min(1, Math.max(0, round(x))), Math.min(1, Math.max(0, round(y)))];
  }

  function onCanvasClick(evt) {
    if (zonesClosed[activeZone]) {
      return;
    }
    zones[activeZone].push(canvasPointFromEvent(evt));
    updateHiddenInput();
    draw();
  }

  function onUndo() {
    if (zonesClosed[activeZone]) {
      return;
    }
    zones[activeZone].pop();
    updateHiddenInput();
    draw();
  }

  function onClose() {
    if (zones[activeZone].length < 3) {
      zoneHintEl.textContent = `${ZONE_LABELS[activeZone]} needs at least 3 points before it can be closed.`;
      return;
    }
    zonesClosed[activeZone] = true;
    updateZoneButtons();
    draw();
  }

  function onEdit() {
    zonesClosed[activeZone] = false;
    updateZoneButtons();
  }

  function onClear() {
    zones[activeZone] = [];
    zonesClosed[activeZone] = false;
    updateHiddenInput();
    updateZoneButtons();
    draw();
  }

  function renderSaveErrors(errors) {
    const list = document.createElement("ul");
    list.className = "error";
    for (const message of errors) {
      const item = document.createElement("li");
      item.textContent = message;
      list.appendChild(item);
    }
    saveResultEl.innerHTML = "";
    saveResultEl.appendChild(list);
  }

  function initZonePicker() {
    document.querySelectorAll(".zone-btn").forEach((btn) => {
      btn.addEventListener("click", () => setActiveZone(btn.dataset.zone));
    });
    document.getElementById("undo-btn").addEventListener("click", onUndo);
    document.getElementById("close-btn").addEventListener("click", onClose);
    document.getElementById("edit-btn").addEventListener("click", onEdit);
    document.getElementById("clear-btn").addEventListener("click", onClear);
    canvas.addEventListener("click", onCanvasClick);
  }

  function loadExistingZones() {
    return fetch("/zones/current")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!data) {
          return;
        }
        for (const name of ZONE_NAMES) {
          const points = data[name] || [];
          zones[name] = points.map(([x, y]) => [x, y]);
          zonesClosed[name] = points.length > 0;
        }
        updateHiddenInput();
        updateZoneButtons();
        draw();
      });
  }

  function loadFrame() {
    return fetch("/zones/frame.jpg", { cache: "no-store" }).then((response) => {
      if (!response.ok) {
        frameMissingEl.classList.remove("hidden");
        editorEl.classList.add("hidden");
        return;
      }
      return response.blob().then((blob) => {
        const url = URL.createObjectURL(blob);
        image.onload = () => {
          canvas.width = image.naturalWidth;
          canvas.height = image.naturalHeight;
          imageLoaded = true;
          URL.revokeObjectURL(url);
          frameMissingEl.classList.add("hidden");
          editorEl.classList.remove("hidden");
          draw();
        };
        image.src = url;
      });
    });
  }

  // `htmx:responseError` fires for any non-2xx response to the `/zones`
  // POST; FastAPI's HTTPException answers with `{"detail": [...]}` JSON,
  // not HTML, so it is rendered here rather than relying on htmx's default
  // (successful-response-only) swap.
  document.body.addEventListener("htmx:responseError", (evt) => {
    let detail;
    try {
      detail = JSON.parse(evt.detail.xhr.response).detail;
    } catch (err) {
      detail = ["save failed"];
    }
    if (!Array.isArray(detail)) {
      detail = [String(detail)];
    }
    renderSaveErrors(detail);
  });

  initZonePicker();
  updateHiddenInput();
  updateZoneButtons();
  // Without this catch the page stays exactly as it loads -- editor hidden,
  // "no frame yet" hidden -- so any unexpected failure in the chain leaves
  // the caregiver a blank page under the heading with nothing to act on.
  // A 503 from `/zones/frame.jpg` is already handled inside `loadFrame`;
  // this is for everything else.
  loadFrame()
    .then(() => loadExistingZones())
    .catch((err) => {
      console.error("zone editor failed to load", err);
      document.getElementById("load-failed").classList.remove("hidden");
      frameMissingEl.classList.add("hidden");
      editorEl.classList.add("hidden");
    });
})();
