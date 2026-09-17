// Homepage "what the software sees" demo (HANDOFF.md V9). Draws the pose
// skeleton, bbox, state badge and timeline over the plain `/sample/sample.mp4`
// using the same coordinate transform as `visualize.py`'s pipeline overlay,
// so this stays visually identical to the reference render without any live
// analysis running in the browser.

(function () {
  "use strict";

  const VIDEO_W = 1280;
  const VIDEO_H = 720;
  const TIMELINE_H = 54;

  const STATE_COLORS = {
    in_bed: "rgb(74, 222, 128)",
    sitting_up: "rgb(250, 204, 82)",
    standing: "rgb(71, 170, 255)",
    walking: "rgb(30, 211, 238)",
    upright: "rgb(71, 170, 255)",
    on_floor: "rgb(255, 95, 95)",
    absent: "rgb(173, 184, 201)",
  };
  const TIMELINE_DEFAULT_COLOR = "rgb(73, 83, 101)";
  const LINE_COLOR = "rgb(39, 236, 208)";
  const JOINT_FILL = "rgb(255, 255, 255)";
  const JOINT_OUTLINE = "rgb(14, 116, 144)";

  const LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
  ];
  const EDGES = [
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 4],
    [3, 5],
    [5, 7],
    [4, 6],
    [6, 8],
  ];

  const els = {
    video: document.getElementById("sample-video"),
    overlay: document.getElementById("sample-overlay"),
    timeline: document.getElementById("sample-timeline"),
  };

  if (!els.video || !els.overlay || !els.timeline) return;

  const overlayCtx = els.overlay.getContext("2d");
  const timelineCtx = els.timeline.getContext("2d");

  let records = [];
  let duration_s = 0;

  function runtimeToSource(x, y) {
    return [x, (y - 0.125) / 0.75];
  }

  function toCanvas(x, y) {
    const [sx, sy] = runtimeToSource(x, y);
    return [sx * VIDEO_W, sy * VIDEO_H];
  }

  function findRecord(t) {
    let match = null;
    for (const record of records) {
      if (record.t_s > t) break;
      match = record;
    }
    return match;
  }

  function drawGatedBadge() {
    overlayCtx.fillStyle = "rgba(15, 24, 39, 0.85)";
    overlayCtx.fillRect(16, 16, 210, 30);
    overlayCtx.fillStyle = "rgb(173, 184, 201)";
    overlayCtx.font = "14px system-ui, sans-serif";
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText("motion gate: skipped", 26, 31);
  }

  function drawStateBadge(state) {
    const label = state ? state.replace(/_/g, " ").toUpperCase() : "TRACKER WARMING";
    const color = state ? STATE_COLORS[state] || "rgb(230, 237, 246)" : "rgb(145, 158, 176)";
    overlayCtx.font = "bold 16px system-ui, sans-serif";
    const textWidth = overlayCtx.measureText(label).width;
    const padding = 12;
    const width = textWidth + padding * 2;
    overlayCtx.fillStyle = "rgba(15, 24, 39, 0.85)";
    overlayCtx.fillRect(16, 16, width, 32);
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(16, 16, width, 32);
    overlayCtx.fillStyle = color;
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText(label, 16 + padding, 32);
  }

  function drawPose(record) {
    const landmarks = record.landmarks || {};
    const points = LANDMARK_NAMES.map((name) => {
      const value = landmarks[name];
      if (!value || value[2] < 0.3) return null;
      return toCanvas(value[0], value[1]);
    });

    overlayCtx.strokeStyle = LINE_COLOR;
    overlayCtx.lineWidth = 4;
    for (const [a, b] of EDGES) {
      if (points[a] && points[b]) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(points[a][0], points[a][1]);
        overlayCtx.lineTo(points[b][0], points[b][1]);
        overlayCtx.stroke();
      }
    }
    for (const point of points) {
      if (!point) continue;
      overlayCtx.beginPath();
      overlayCtx.arc(point[0], point[1], 4, 0, Math.PI * 2);
      overlayCtx.fillStyle = JOINT_FILL;
      overlayCtx.fill();
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeStyle = JOINT_OUTLINE;
      overlayCtx.stroke();
    }

    if (record.bbox) {
      const [x1, y1] = toCanvas(record.bbox[0], record.bbox[1]);
      const [x2, y2] = toCanvas(record.bbox[2], record.bbox[3]);
      overlayCtx.strokeStyle = LINE_COLOR;
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }

  function drawOverlay() {
    overlayCtx.clearRect(0, 0, VIDEO_W, VIDEO_H);
    const t = els.video.currentTime;
    const record = findRecord(t);
    if (!record || record.gated) {
      drawGatedBadge();
      return;
    }
    drawPose(record);
    drawStateBadge(record.state);
  }

  function drawTimeline() {
    const width = els.timeline.width;
    const height = els.timeline.height;
    timelineCtx.clearRect(0, 0, width, height);
    timelineCtx.fillStyle = "rgb(15, 24, 39)";
    timelineCtx.fillRect(0, 0, width, height);

    if (records.length === 0) return;
    const barY = 20;
    const barH = 12;
    for (let pixel = 0; pixel < width; pixel += 1) {
      const index = Math.min(records.length - 1, Math.floor((pixel / width) * records.length));
      const state = records[index].state;
      timelineCtx.strokeStyle = state ? STATE_COLORS[state] || TIMELINE_DEFAULT_COLOR : TIMELINE_DEFAULT_COLOR;
      timelineCtx.beginPath();
      timelineCtx.moveTo(pixel, barY);
      timelineCtx.lineTo(pixel, barY + barH);
      timelineCtx.stroke();
    }

    const t = els.video.currentTime;
    const playheadX = duration_s > 0 ? (t / duration_s) * width : 0;
    timelineCtx.strokeStyle = "rgb(255, 255, 255)";
    timelineCtx.lineWidth = 3;
    timelineCtx.beginPath();
    timelineCtx.moveTo(playheadX, barY - 5);
    timelineCtx.lineTo(playheadX, barY + barH + 5);
    timelineCtx.stroke();
  }

  function draw() {
    drawOverlay();
    drawTimeline();
  }

  function seekFromEvent(event) {
    const rect = els.timeline.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    if (duration_s > 0) els.video.currentTime = fraction * duration_s;
  }

  function attachVideoFrameLoop() {
    if (typeof els.video.requestVideoFrameCallback === "function") {
      const onFrame = () => {
        draw();
        els.video.requestVideoFrameCallback(onFrame);
      };
      els.video.requestVideoFrameCallback(onFrame);
    } else {
      const onFrame = () => {
        draw();
        requestAnimationFrame(onFrame);
      };
      requestAnimationFrame(onFrame);
    }
  }

  async function main() {
    els.overlay.width = VIDEO_W;
    els.overlay.height = VIDEO_H;
    els.timeline.width = VIDEO_W;
    els.timeline.height = TIMELINE_H;

    const response = await fetch("/sample/sample.json");
    if (!response.ok) return;
    records = await response.json();
    duration_s = records.length ? records[records.length - 1].t_s : 0;

    let dragging = false;
    els.timeline.addEventListener("mousedown", (event) => {
      dragging = true;
      seekFromEvent(event);
    });
    window.addEventListener("mousemove", (event) => {
      if (dragging) seekFromEvent(event);
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });
    els.timeline.addEventListener(
      "touchstart",
      (event) => seekFromEvent(event.touches[0]),
      { passive: true },
    );
    els.timeline.addEventListener(
      "touchmove",
      (event) => seekFromEvent(event.touches[0]),
      { passive: true },
    );

    els.video.src = "/sample/sample.mp4";
    draw();
    attachVideoFrameLoop();
  }

  main();
})();
