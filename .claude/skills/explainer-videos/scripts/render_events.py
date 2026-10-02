"""Events in, events out: silent 62-second explainer from a captured agent run.

Usage: render_events.py [case.json] [out.mp4|out.png] [still-second]
CLIPS=0 freezes the thumbnails; moving pre-extracted frames are on by default.
Captured facts are selected in load_case; editorial captions and timings live here.
Capture omissions: Frame and IDLE are architectural labels supplied by the brief;
the capture records rejection, but not its word-level reason. The highlighted
conjunction is extracted from the draft, rather than retyping any captured text.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion import (WORK, Canvas, W, H, S, FPS, GROUND, PAPER, LINE, LINE_SOFT,
                    INK, GRAY, FAINT, WARM, WARM_BG, WARM_INK, COOL, COOL_BG,
                    ALERT, ALERT_BG, ALERT_PILL, ALERT_INK, mono, sans, serif,
                    serif_i, sc, clamp, prog, ease, mix, wrap, bezier, chip,
                    point_at, fade_all, encode)

DUR = 62.0
CAM = (60, 455, 345, 345)
STA = (447, 470, 305, 330)
AGT = (800, 420, 470, 470)
LLM = (822, 690, 426, 178)
VOI = (1350, 370, 490, 215)
ALR = (1350, 665, 490, 205)
INS = [(440, 235, 205, 122), (663, 235, 205, 122),
       (886, 235, 190, 122), (1094, 235, 215, 122)]
PORTS = [880, 975, 1070, 1190]  # where the input wires meet the agent's top edge
OUT_DIR = WORK / "event-system-2026-10-01"
# Pre-extracted 30 fps JPEG frames of the bedroom-sample-02 pose demo clips (walk, floor).
CLIP_WORK = Path(os.environ.get("EXPLAINER_CLIPS", WORK / "pipeline-case-2026-09-18/video-02/_clip_frames"))
CLIPS = {"up": (1.8, 0.5), "floor": (2.6, 0.75)}
_STATE = {}


def load_case(path):
    with open(path) as f:
        log = json.load(f)
    inputs = [e for e in log if e["kind"] == "input"]
    people = [e for e in inputs if "state" in e["data"]]
    utter = [e for e in inputs if "text" in e["data"]]
    interp = [e for e in log if e["kind"] == "llm" and
              e["call"] == "Interpretation" and e.get("output")]
    comp = next(e for e in log if e["kind"] == "llm" and
                e["call"] == "Composition" and e.get("output"))
    phases = [e for e in log if e["kind"] == "phase"]
    says = [e for e in log if e["kind"] == "output" and "strategy" in e["data"]]
    replies = [next(e for e in says if e["at"] == u["at"]) for u in utter]
    escalation = next(e for e in says if e["at"] == phases[-1]["at"])
    notify = next(e for e in log if e["kind"] == "output" and "repeat_until_ack" in e["data"])
    rejected = next(e for e in log if e["kind"] == "output" and
                    e["data"].get("kind") == "compose" and e["data"].get("ok") is False)
    decisions = [json.loads(e["data"]["detail"]) for e in log if
                 e["kind"] == "output" and e["data"].get("kind") == "decision"]
    safety = next(d for d in decisions if d.get("phase") == phases[-1]["data"]["phase"])
    profile = comp["input"]["profile"]
    # Derive the conjunction in the draft from its comma-delimited clause.
    veto_word = comp["output"]["text"].split(",")[-1].strip().split()[0]
    held = int((datetime.fromisoformat(phases[-1]["at"]) -
                datetime.fromisoformat(people[-1]["at"])).total_seconds())
    events = {"state": people[0]["event"], "utter": utter[0]["event"],
              "say": replies[0]["event"], "notify": notify["event"]}
    return dict(up=people[0]["data"], floor=people[-1]["data"],
                utter=[u["data"]["text"] for u in utter],
                reply=[e["data"] for e in replies] + [escalation["data"]],
                interpret=[e["output"] for e in interp], draft=comp["output"]["text"],
                model=comp["model"].split(":")[0], profile=profile,
                calming=profile["calming_things"][0],
                caregiver=f'{profile["caregiver_name"]} ({profile["caregiver_relationship"]})',
                phases=[e["data"]["phase"] for e in phases],
                goals=[phases[0]["data"]["goal"], phases[-1]["data"]["goal"]],
                clocks=[e["at"][11:16] for e in (people[0], utter[-1], people[-1])],
                hour=int(people[0]["at"][11:13]), time_words=comp["input"]["time_words"],
                held=held, rule=safety["trigger"].split("_")[0].removeprefix("rule"),
                notify=notify["data"], events=events, veto_word=veto_word,
                rejection=rejected["data"]["detail"])


def fit_mono(text, width, size=13):
    while mono(size, "Medium").getlength(text) / S > width:
        size -= 0.25
    return mono(size, "Medium")


def arrival(t, at, span=0.5):
    return ease(prog(t, at, at + span))


def travelling(cv, pts, t, at, label, col=INK, bg=PAPER, stop=1.0):
    """Slide an event pill along `pts`, then hold it; `stop` < 1 parks it mid-wire
    so a pill on a short gap never covers the next box's text."""
    if at <= t <= at + 2.2:
        p = prog(t, at, at + 0.8)
        a = arrival(t, at, 0.15) * (1 - arrival(t, at + 1.9, 0.3))
        x, y = point_at(pts, ease(p) * stop)
        chip(cv, x, y, label, a, col, bg)


@lru_cache(maxsize=24)
def clip_frame(key, index):
    with Image.open(_STATE["frames"][key][index]) as im:
        return im.convert("RGB")


def thumbnail(key, t):
    offset, speed = CLIPS[key]
    start = 2.5 if key == "up" else 38.5
    # Replay the walking loop for the second conversational beat.
    elapsed = max(0, t - start)
    if key == "up":
        elapsed %= 8.0
    c = offset + elapsed * speed if _STATE["moving"] else offset
    if key == "floor":
        # Keep the established seated pose; the source later stands and fades.
        c = min(c, 9.0)
    frames = _STATE["frames"][key]
    return clip_frame(key, min(len(frames) - 1, int(c * FPS)))


def draw_frame(t, C):
    cv = Canvas()
    beat = 0 if t < 20.5 else 1 if t < 38.5 else 2
    intro = arrival(t, 0.2, 1)
    floor = arrival(t, 39.4, 0.7)
    escalated = arrival(t, 45.0, 0.6)
    ghost = arrival(t, 0.6, 1.2)
    headings = ["The model reads. The agent answers.",
                "The model drafts. The agent checks.",
                "On the floor: a rule, not a model."]
    for i, (start, end) in enumerate(((0.2, 20.5), (20.5, 38.5), (38.5, 63))):
        a = arrival(t, start, 0.65) * (1 - arrival(t, end - 0.35, 0.35))
        if a:
            cv.text(77, 84, headings[i], serif(46), mix(INK, GROUND, a))
    cv.text(77, 58, f"NIGHT COMPANION  ·  {C['clocks'][beat]}",
            mono(15, "Medium"), mix(FAINT, GROUND, intro))
    for x, y, w, h in (CAM, STA, AGT, VOI, ALR):
        cv.rrect(x, y, w, h, 12, fill=mix(PAPER, GROUND, ghost),
                 outline=mix(LINE_SOFT, GROUND, ghost))

    # Context cards feed the agent, which hands them to its model; dashed, no heads.
    for i, (ix, iy, iw, ih) in enumerate(INS[:3]):
        a = arrival(t, 6 + i * 0.5) * (1 - 0.7 * floor)
        if i == 1:
            a *= arrival(t, 21.0)
        port = PORTS[i]
        pts = bezier((ix + iw / 2, iy + ih + 6), (ix + iw / 2, iy + ih + 40),
                     (port, AGT[1] - 36), (port, AGT[1] - 6))
        cv.polyline(pts, mix(LINE, GROUND, a), 1.5, dash=(4, 6), frac=a)

    # Live events follow service boundaries; the alert always traverses AGT.
    framewire = [(CAM[0] + CAM[2] + 6, 635), (STA[0] - 6, 635)]
    statewire = [(STA[0] + STA[2] + 6, 615), (AGT[0] - 6, 615)]
    utterwire = bezier((1201.5, 363), (1201.5, 385), (PORTS[3], 392), (PORTS[3], AGT[1] - 6))
    out1 = bezier((1270, 559), (1310, 559), (1308, 478), (1350, 478))
    out2 = bezier((1270, 638), (1310, 638), (1306, 768), (1350, 768))
    cv.wire(framewire, mix(FAINT, GROUND, arrival(t, 2.8)), frac=arrival(t, 2.8))
    cv.wire(statewire, mix(WARM if beat == 2 else FAINT, GROUND, arrival(t, 4.4)),
            frac=arrival(t, 4.4))
    cv.wire(utterwire, mix(COOL, GROUND, arrival(t, 8) * (1 - 0.8 * floor)),
            frac=arrival(t, 8))
    cv.wire(out1, mix(WARM if beat == 2 else FAINT, GROUND, arrival(t, 13.8)),
            frac=arrival(t, 13.8))
    if beat == 2:
        cv.wire(out2, mix(ALERT, GROUND, arrival(t, 46.0)), frac=arrival(t, 46.0))

    # Camera, preserving the hub's full-frame moving thumbnail and captions.
    a = arrival(t, 2.5, 0.6)
    x, y, w, h = CAM
    cv.rrect(x, y, w, h, 12, fill=PAPER, outline=mix(LINE, GROUND, a))
    cv.text(x + 22, y + 22, "CAMERA  ·  perceive", mono(13, "Medium"), mix(FAINT, PAPER, a))
    tw, th = w - 44, (w - 44) * 9 / 16
    if a:
        cv.paste(thumbnail("up", t), x + 22, y + 52, tw, th, 8, a)
        if beat == 2:
            cv.paste(thumbnail("floor", t), x + 22, y + 52, tw, th, 8, arrival(t, 38.5, 0.8))
    cv.text(x + 22, y + 52 + th + 18, "pose + vision model · on device", mono(13), mix(GRAY, PAPER, a))
    cv.text(x + 22, y + 52 + th + 42, "frames never leave the room", mono(13), mix(FAINT, PAPER, a))

    x, y, w, h = STA
    a = arrival(t, 3.4, 0.6)
    if floor:
        cv.rrect(x - 6, y - 6, w + 12, h + 12, 16, fill=mix(WARM_BG, GROUND, floor))
    cv.rrect(x, y, w, h, 12, fill=PAPER, outline=mix(mix(WARM, LINE, floor), GROUND, a), width=1.5 + floor)
    cv.text(x + 22, y + 22, C["events"]["state"].upper(), mono(13, "Medium"), mix(FAINT, PAPER, a))
    p = C["floor"] if t >= 39.4 else C["up"]
    title = "On the ground" if t >= 39.4 else "Up and moving"
    cv.text(x + 22, y + 50, title, sans(25, "SemiBold"), mix(INK, PAPER, a))
    chip(cv, x + 22, y + 108, f'{p["state"]} · {p["confidence"]:.2f}', a,
         WARM_INK if floor else GRAY, WARM_BG if floor else (235, 234, 229), size=15, anchor="la")
    note = wrap(f'“{p["scene_note"]}”', sans(19), w - 44)
    cv.lines(x + 22, y + 140, note, sans(19), mix(GRAY, PAPER, a), 27,
             chars=max(0, t - (40.0 if floor else 4.0)) * 50)

    # Inputs fall into place; history only appears in the second beat.
    vals = [f'“{C["calming"]}”', f'“{C["utter"][0]}”',
            f'goal {C["goals"][0]} · {C["hour"]} o’clock at night',
            f'“{C["utter"][min(beat, 1)]}”']
    for i, ((x, y, w, h), label, val) in enumerate(zip(INS, ("PROFILE", "HISTORY", "TONIGHT", "VOICE INPUT"), vals)):
        a = arrival(t, 6 + i * 0.5)
        if i == 1:
            a *= arrival(t, 21.0)
        if not a:
            continue
        drop = -18 * (1 - a)
        y += drop
        vis = a * (1 - 0.65 * floor)
        highlight = i == 0 and 26.4 <= t < 38.5
        bg = WARM_BG if highlight else PAPER
        cv.rrect(x, y, w, h, 8, fill=mix(bg, GROUND, vis),
                 outline=mix(WARM if highlight else COOL if i == 3 else LINE, GROUND, vis),
                 width=1.8 if i == 3 or highlight else 1.4, dash=i != 3)
        cv.text(x + 14, y + 12, label, mono(11.5, "Medium"), mix(FAINT, GROUND, vis))
        cv.lines(x + 14, y + 38, wrap(val, sans(16.5, "Medium"), w - 28),
                 sans(16.5, "Medium"), mix(INK if i in (0, 3) else GRAY, GROUND, vis), 22)
        if floor:
            # Separate baseline keeps the editorial badge clear of long labels.
            cv.text(x + w - 12, y + h + 8, "not used", mono(11.5, "Medium"), mix(WARM_INK, GROUND, floor), "ra")

    # Hero: persistent deterministic state machine, with a model inside it.
    x, y, w, h = AGT
    a = arrival(t, 4.8, 0.6)
    cv.rrect(x, y, w, h, 14, fill=PAPER, outline=mix(INK, LINE_SOFT, a), width=2.5)
    cv.text(x + 22, y + 20, "AGENT SERVICE  ·  deterministic code", mono(13, "Medium"), mix(FAINT, PAPER, a))
    cv.text(x + 22, y + 45, "State machine", sans(30, "SemiBold"), mix(INK, PAPER, a))
    phase_names = ["IDLE"] + C["phases"]
    phase_at = [4.8, 5.2, 12.8, 45.0]
    active = 0 if t < 5.2 else 1 if t < 12.8 else 2 if t < 45 else 3
    # A filled capsule slides along the rail on each case phase transition.
    rail_x, widths, gap = x + 22, [54, 109, 90, 110], 9
    positions = [rail_x]
    for width in widths[:-1]:
        positions.append(positions[-1] + width + gap)
    transition = arrival(t, phase_at[active], 0.55)
    prev = max(0, active - 1)
    hx = positions[prev] + (positions[active] - positions[prev]) * transition
    hw = widths[prev] + (widths[active] - widths[prev]) * transition
    cv.rrect(hx, y + 92, hw, 28, 14, fill=mix(ALERT_BG if active == 3 else COOL_BG, PAPER, a),
             outline=mix(ALERT if active == 3 else INK, PAPER, a), width=1)
    for i, label in enumerate(phase_names):
        cv.text(positions[i] + widths[i] / 2, y + 99, label, mono(11.5, "Medium"),
                mix(ALERT_INK if active == 3 and i == 3 else INK if i == active else FAINT, PAPER, a), "ma")
    goal = C["goals"][1 if t >= 45 else 0]
    cv.text(x + 22, y + 137, f'goal      {goal}', mono(16), mix(GRAY, PAPER, a))
    if t >= 12.8:
        reply = C["reply"][2 if t >= 45 else min(beat, 1)]["strategy"]
        cv.text(x + 22, y + 165, f'reply     {reply}', mono(16), mix(GRAY, PAPER, arrival(t, 12.8)))
    rule = f'rule {C["rule"]} · floor'
    for gx, label, lit, color in ((x + 22, "check draft", 30.2 <= t < 38.5, WARM),
                                 (x + 147, "veto", False, ALERT),
                                 (x + 213, rule, t >= 42.0, WARM)):
        chip(cv, gx, y + 211, label, a, color if lit else FAINT,
             ALERT_BG if lit and color == ALERT else WARM_BG if lit else PAPER,
             size=12, anchor="la")
    if t >= 42:
        timer = prog(t, 42, 44.5)
        cx, cy, r = 1169, 631, 12
        cv.d.ellipse((sc(cx-r), sc(cy-r), sc(cx+r), sc(cy+r)), outline=LINE_SOFT, width=sc(2))
        cv.d.arc((sc(cx-r), sc(cy-r), sc(cx+r), sc(cy+r)), -90, -90 + 360 * timer, fill=WARM, width=sc(2.5))
        cv.text(1245, 651, f'held {round(C["held"] * timer)} s', mono(12, "Medium"), WARM_INK, "ra")

    # Short internal handoffs are local helper calls, not service publications.
    # They occupy a dedicated strip between the checks and the nested model.
    if beat < 2:
        call_at, return_at = (9.0, 12.0) if beat == 0 else (23.0, 25.2)
        down = [(865, 668), (865, 689)]
        up = [(1156, 689), (1156, 668)]
        da = arrival(t, call_at) * (1 - arrival(t, return_at + 1, 0.5))
        ua = arrival(t, return_at) * (1 - arrival(t, return_at + 1.8, 0.5))
        if beat == 1 and 28.8 <= t < 30.4:
            ua = arrival(t, 28.8) * (1 - arrival(t, 29.8))
        cv.wire(down, mix(COOL, PAPER, da), width=1.5)
        cv.text(883, 667, "text in", mono(11), mix(COOL, PAPER, da))
        cv.wire(up, mix(COOL, PAPER, ua), width=1.5)
        cv.text(1139, 667, "draft" if beat == 1 and t >= 28.8 else "intent", mono(11), mix(COOL, PAPER, ua), "ra")

    x, y, w, h = LLM
    dim = 1 - 0.65 * floor
    cv.rrect(x, y, w, h, 8, fill=mix(COOL_BG, PAPER, dim * a),
             outline=mix(COOL, PAPER, dim * a), dash=True)
    tag = f'LOCAL LLM  ·  {C["model"]}  ·  advises, never decides'
    cv.text(x + 16, y + 14, tag, fit_mono(tag, w - 32), mix(GRAY, PAPER, dim * a))
    if beat == 2:
        cv.text(x + 20, y + 72, "not called", mono(15, "Medium"), mix(WARM_INK, PAPER, floor))
    else:
        it = C["interpret"][beat]
        start, done = (9.8, 11.5) if beat == 0 else (23.6, 25.0)
        if start <= t < done:
            cv.text(x + 16, y + 47, "thinking " + "·" * (1 + int(t * 4) % 3), mono(14), GRAY)
        elif t >= done:
            label = f'interpret → {it["intent"]} · distress {it["distress"]}'
            cv.text(x + 16, y + 47, label, fit_mono(label, w-32, 13), mix(INK, COOL_BG, arrival(t, done)))
        if beat == 1 and t >= 26.0:
            cv.text(x + 16, y + 72, "draft →", mono(12), GRAY)
            f = serif_i(18)
            lines = wrap(C["draft"], f, w - 32)
            assert len(lines) <= 3, "draft must fit three lines"
            chars = max(0, t - 26) * (len(C["draft"]) + 4) / 2.9
            rejected = arrival(t, 30.9, 0.3)
            for i, line in enumerate(lines):
                yy = y + 93 + i * 21
                visible = line[:max(0, int(chars))]
                chars -= len(line) + 1
                if C["veto_word"] in line.split() and t >= 30.2:
                    before = line.split(C["veto_word"])[0]
                    wx = x + 16 + f.getlength(before)/S
                    ww = f.getlength(C["veto_word"])/S
                    cv.rrect(wx-2, yy, ww+4, 19, 2, fill=ALERT_PILL)
                    cv.polyline([(wx, yy+19), (wx+ww, yy+19)], ALERT, 2)
                cv.text(x + 16, yy, visible, f, mix(INK, COOL_BG, 1 - 0.62 * rejected))
                if rejected:
                    cv.polyline([(x + 16, yy+9), (x+16+f.getlength(line)/S, yy+9)], mix(ALERT, COOL_BG, rejected*.5), 1)
            if rejected:
                # Reserve the bottom strip for the stamp; no draft text underneath.
                chip(cv, x+w/2, y+162, f'rejected · contains "{C["veto_word"]}"', rejected,
                     ALERT_INK, ALERT_BG, size=11)

    # Bedside quotes reveal word by word, then hold well beyond two seconds.
    x, y, w, h = VOI
    vb = beat
    quote_at = [14.7, 32.7, 47.8][vb]
    va = arrival(t, quote_at - 0.3)
    stale = 1.0
    if not va and vb:
        # Until the next line, the last one stays up, dimmed, so the box never goes blank.
        vb, va, stale = vb - 1, 1.0, 0.45
        quote_at = [14.7, 32.7, 47.8][vb]
    if va:
        cv.rrect(x, y, w, h, 12, fill=mix(COOL_BG, PAPER, va))
        source = ["approved line, picked by the agent", "caregiver's fallback", "fixed safety phrase"][vb]
        tag = f'BEDSIDE VOICE  ·  {source}'
        cv.text(x+26, y+22, tag, fit_mono(tag, w-52), mix(FAINT, COOL_BG, va))
        q = f'“{C["reply"][vb]["text"]}”'
        size = 30 if vb < 2 else 40
        words = max(0, int((t - quote_at) * len(q.split()) / 2.5))
        shown = " ".join(q.split()[:words])
        cv.lines(x+26, y+57, wrap(shown, serif_i(size), w-52), serif_i(size), mix(INK, COOL_BG, stale), size*1.18)
        speaking = quote_at <= t < quote_at + 2.5 and stale == 1.0
        for k in range(26):
            amp = (0.25 + .75 * abs(math.sin(t*7.3+k*.9)*math.sin(t*2.1+k*.37))) if speaking else .12
            bh = 4 + 22*amp
            cv.rrect(x+26+k*9, y+h-36-bh/2, 4.5, bh, 2,
                     fill=mix(COOL, COOL_BG, va*(1 if speaking else .5)))

    # Phone keeps the hub's pop, alert palette and buzz ring.
    x, y, w, h = ALR
    alert = arrival(t, 47.0) if beat == 2 else 0
    if alert:
        pop = prog(t, 47, 47.5)
        grow = 1 + .04*math.sin(math.pi*pop)
        gx, gy = w*(grow-1)/2, h*(grow-1)/2
        cv.rrect(x-gx, y-gy, w*grow, h*grow, 12, fill=mix(ALERT_BG, PAPER, alert))
    pa = arrival(t, 5.5)
    cv.text(x+26, y+22, f'CAREGIVER PHONE  ·  {C["caregiver"]}', mono(13, "Medium"), mix(FAINT, PAPER, pa))
    if alert:
        n = C["notify"]
        cv.text(x+26, y+52, n["title"], sans(30,"SemiBold"), mix(INK, ALERT_BG, alert))
        cv.text(x+26, y+96, n["body"], sans(20), mix(GRAY, ALERT_BG, alert))
        repeat = "repeats until seen" if n["repeat_until_ack"] else "once"
        chip(cv, x+26, y+155, f'{n["level"]} · {repeat}', alert, ALERT_INK, ALERT_PILL, 15, anchor="la")
        ring = prog(t, 47.2, 48.4)
        if 0 < ring < 1:
            r = 10 + 30*ring
            cx, cy = x+w-40, y+40
            cv.d.ellipse((sc(cx-r),sc(cy-r),sc(cx+r),sc(cy+r)), outline=mix(ALERT,ALERT_BG,1-ring),width=sc(2))
        cv.d.ellipse((sc(x+w-46),sc(y+34),sc(x+w-34),sc(y+46)),fill=mix(ALERT,ALERT_BG,alert))
    else:
        cv.text(x+26, y+54, "No alert: handled at the bedside", sans(21), mix(FAINT,PAPER,pa))

    # Pills are drawn last so short service gutters still carry readable events.
    for at in (2.8, 21.2, 39.1):
        travelling(cv, framewire, t, at, "Frame", COOL, COOL_BG, stop=0.5)
    for at in (4.4, 40.5):
        travelling(cv, statewire, t, at, C["events"]["state"], WARM_INK if beat==2 else INK, WARM_BG if beat==2 else PAPER, stop=0.5)
    for at in (8.1, 22.0):
        travelling(cv, utterwire, t, at, C["events"]["utter"], COOL, COOL_BG)
    for at in (13.8, 31.8, 46.9):
        travelling(cv, out1, t, at, C["events"]["say"], WARM_INK if beat==2 else INK, stop=0.55)
    travelling(cv, out2, t, 46.1, C["events"]["notify"], ALERT_INK, ALERT_BG, stop=0.55)

    caption = "every arrow is one event on a Redis stream · services never call each other"
    cv.text(1035, 911, caption, mono(13), mix(FAINT,GROUND,a), "ma")
    end = arrival(t, 58.5, .8)
    if end:
        cv.text(960, 948, "Safety lives in deterministic code. The model reads and drafts; it never decides.",
                sans(24), mix(INK,GROUND,end), "ma")
        cv.text(960, 986, "Its rules trace to published dementia-care guidance, checked by a person, "
                "and to the family's own phrases.", sans(18), mix(GRAY,GROUND,arrival(t, 59.2, .8)), "ma")
    cv.text(77,1030,f"Real run · today's agent code + local {C['model']} · the person is scripted · nothing leaves the device",
            mono(13),mix(FAINT,GROUND,intro))
    cv.text(1843,1030,"not a medical device",mono(13),mix(FAINT,GROUND,intro),"ra")
    return fade_all(cv.final(),t,DUR)


def _init(case_path):
    _STATE["C"] = load_case(case_path)
    _STATE["moving"] = os.environ.get("CLIPS", "1") != "0"
    _STATE["frames"] = {key: sorted((CLIP_WORK / key).glob("*.jpg")) for key in CLIPS}
    for key, frames in _STATE["frames"].items():
        if not frames:
            raise FileNotFoundError(f"Missing thumbnail frames: {CLIP_WORK / key}")


def _render(i):
    return draw_frame(i/FPS, _STATE["C"]).tobytes()


def main():
    case = sys.argv[1] if len(sys.argv)>1 else str(OUT_DIR/"case.json")
    out = Path(sys.argv[2]) if len(sys.argv)>2 else OUT_DIR/"events-explainer-30fps.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        _init(case)
        draw_frame(float(sys.argv[3]) if len(sys.argv)>3 else 19, _STATE["C"]).save(out)
    else:
        encode(_render, int(DUR*FPS), out, init=_init, initargs=(case,),
               workers=int(os.environ.get("WORKERS", "8")))


if __name__ == "__main__":
    main()
