"""Silent scene_lab loop explainer. All run evidence is loaded from facts JSON.

Usage: render_loop.py [facts.json] [out.mp4|out.png] [still-second]
Authored explanatory labels are separate from captured evidence. The approval cue
illustrates the owner gate; the facts file does not record an approval event.
"""
from __future__ import annotations
import json
import math
import re
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion import (Canvas, mono, sans, serif, serif_i, GROUND, PAPER, LINE,
    LINE_SOFT, INK, GRAY, FAINT, WARM, WARM_BG, WARM_INK, COOL, COOL_BG,
    ALERT, ALERT_BG, ALERT_INK, S, FPS, sc, prog, ease, mix, wrap,
    bezier, chip, point_at, fade_all, encode)

DUR = 74.0
STARTS = [0, 5.5, 19.5, 30, 43, 50.5, 60, 69.5]
HEADINGS = ['Simulated nights', '1 · Claude directs a night',
    '2 · Every night is checked', '3 · Opus triages the night',
    '4 · A person decides', '4 · Who says what is correct',
    '5 · The fix lands, and stays fixed', 'Simulated nights']
# Loop station shown for each panel; the "what is correct" panel stays on station 4.
STATION = [0, 1, 2, 3, 4, 4, 5, 0]
STATION_START = {1: 5.5, 2: 19.5, 3: 30, 4: 43, 5: 60}
OUTRO = STARTS[-1]
TITLES = ['Simulate a night', 'Check every trace', 'Triage', 'Owner decides', 'Fix, then run again']
F = {}


def clean(s):
    return re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', s).replace('`', '')


def sentences(s):
    # Split on sentence boundaries, preserving numeric times and file line references.
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', s)


def pretty(s):
    return s.replace('_', ' ').replace('-', ' ')


def ellipsize(s, f, width):
    if f.getlength(s) / S <= width:
        return s
    while s and f.getlength(s + '…') / S > width:
        s = s[:-1]
    return s.rstrip() + '…'


def words(s, t, start, duration=.8):
    ws = s.split()
    return ' '.join(ws[:int(len(ws) * prog(t, start, start + duration))])


def txt(cv, x, y, s, f, width, lh, col=INK, t=None, start=0, duration=.8):
    if t is not None:
        s = words(s, t, start, duration)
    ls = wrap(s, f, width)
    cv.lines(x, y, ls, f, col, lh)
    return len(ls) * lh


def card(cv, x, y, w, h, border=LINE_SOFT, fill=PAPER, width=1.5):
    cv.rrect(x, y, w, h, 12, fill=fill, outline=border, width=width)


def pill(cv, x, y, s, fg=GRAY, bg=PAPER, size=13, a=1):
    return chip(cv, x, y, s, a=a, fg=fg, bg=bg, size=size, anchor='la')


def circle(cv, x, y, r, col, outline=None):
    cv.d.ellipse((sc(x-r), sc(y-r), sc(x+r), sc(y+r)), fill=col, outline=outline)


def ring_point(lap):
    a = -math.pi/2 + lap * math.tau
    return 440 + 280*math.cos(a), 590 + 280*math.sin(a)


RING = [ring_point(i/360) for i in range(361)]


def loop_map(cv, t):
    reveal = ease(prog(t, .35, 3.4))
    cv.polyline(RING, LINE_SOFT, 2, frac=reveal)
    station = STATION[max(i for i, s in enumerate(STARTS) if t >= s)] if t < OUTRO else 5
    if t >= OUTRO:
        lap = .8 + ease(prog(t, OUTRO, OUTRO + 3.1))
        active = int((lap % 1)*5 + .5) % 5
    else:
        active = station-1
        lap = max(0, active)/5
        if station > 1:
            lap = (active-1 + ease(prog(t, STATION_START[station], STATION_START[station]+.8)))/5
    if station:
        cv.polyline(RING, FAINT, 2.3, frac=min(1, lap))
    for i in range(5):
        p = ring_point((i+.48)/5)
        q = ring_point((i+.46)/5)
        if reveal > (i+.48)/5:
            cv.arrow_head(p, q, LINE, 10)
    if station:
        x,y = ring_point(lap)
        # Cards are offset from the ring so the travelling dot stays visible.
        circle(cv,x,y,16,COOL_BG)
        circle(cv,x,y,7,COOL)
    roles = [f"Opus directs · Sonnet plays Jean", f"{len(F['invariants'])} checks, no labels",
             'Opus · throwaway worktree', 'a person', 'Claude Code commits']
    for i, (title, role) in enumerate(zip(TITLES, roles)):
        a = ease(prog(t, .55+i*.42, 1.05+i*.42))
        lit = (active == i and station > 0)
        x, y = ring_point(i/5)
        w = max(230, sans(19,'SemiBold').getlength(title)/S+40, mono(12).getlength(role)/S+28)
        y += [-62, -66, 62, 62, -66][i] + (1-a)*12
        card(cv, x-w/2, y-38, w, 76,
             mix(INK if lit else LINE, GROUND, a), mix(PAPER, GROUND, a), 2 if lit else 1.3)
        alpha = a*(1 if lit else .45)
        cv.text(x, y-18, title, sans(19,'SemiBold'), mix(INK,PAPER,alpha), 'ma')
        cv.text(x, y+11, role, mono(12), mix(GRAY,PAPER,alpha), 'ma')
    a = ease(prog(t, 2.4, 3.4))
    cv.text(440, 528, 'DIRECTOR', mono(13,'Medium'), mix(FAINT,GROUND,a), 'ma')
    cv.text(440, 557, F['roles']['director'], sans(30,'SemiBold'), mix(INK,GROUND,a), 'ma')
    cv.text(440, 605, 'picks each scene,', sans(17), mix(GRAY,GROUND,a), 'ma')
    cv.text(440, 632, 'triages each night', sans(17), mix(GRAY,GROUND,a), 'ma')


def intro(cv, t):
    card(cv, 890, 325, 940, 365)
    cv.text(920, 355, 'SIMULATED NIGHTS · REAL AGENT', mono(13,'Medium'), FAINT)
    txt(cv,920,410,'Nobody can test a night companion at 3 a.m. with a real person in the room. So Claude plays one, the real agent answers, and every night is checked and triaged automatically.',sans(26),820,38,t=t,start=.5,duration=1.7)


def simulate(cv, t):
    scene = F['scene']
    card(cv,890,200,940,220)
    cv.text(914,221,'SCENE CARD  ·  written by the director',mono(13,'Medium'),FAINT)
    cv.text(914,250,scene['id'],mono(18),INK)
    x=914
    for s in [scene['category'],scene['persona_tag'],*scene['stressors']]:
        x += pill(cv,x,301,pretty(s),GRAY,PAPER,12)+10
    txt(cv,914,328,scene['summary'].split('. ')[0]+'.',sans(18),890,25,GRAY)
    rationale = next(s for s in sentences(scene['director_rationale']) if s.startswith('To isolate'))
    txt(cv,914,374,rationale,serif_i(20),880,26,INK,t=t,start=.7,duration=.65)
    # Three nodes, wires, and events on the real agent path.
    nodes=[(890,478,224,106),(1240,478,282,106),(1638,478,192,106)]
    for x,y,w,h in nodes:
        card(cv,x,y,w,h)
    cv.text(906,495,'JEAN',mono(13,'Medium'),FAINT)
    cv.text(906,527,F['roles']['person'],sans(21,'SemiBold'),INK)
    cv.text(1257,495,'REAL STACK',mono(13,'Medium'),FAINT)
    txt(cv,1257,525,f"listen · agent + {F['agent_model'].split(':')[0]} · embodiment",sans(18),252,25)
    cv.text(1654,495,'VIRTUAL PAGE',mono(12,'Medium'),FAINT)
    txt(cv,1654,526,'plays the speech',sans(18),165,24)
    p1=[(1119,533),(1234,533)]; p2=[(1527,533),(1632,533)]
    cv.wire(p1,FAINT,1.7); cv.wire(p2,FAINT,1.7)
    ret=bezier((1734,590),(1734,645),(1002,645),(1002,590))
    cv.wire(ret,LINE,1.5)
    for i,label in enumerate(['PersonState','audio']):
        half=(mono(10,'Medium').getlength(label)/S+22)/2
        x,y=point_at(p1,.5)
        x += (57-half)*math.sin(t*1.2+i*math.pi)
        chip(cv,x,y-22+i*44,label,size=10,fg=COOL)
    x,y=point_at(p2,.5); x += 29*math.sin(t*1.3); chip(cv,x,y,'Say',size=11,fg=COOL)
    x,y=point_at(ret,(t*.16)%1); chip(cv,x,y,'playback ended',size=11)
    cv.text(910,650,"real speech in and out: Piper voices, Whisper, the agent's own model",mono(13),FAINT)
    # A clipped scrolling viewport keeps whole chat rows inside the ticker.
    card(cv,890,685,940,275)
    rows=[F['transcript'][i] for i in [0,2,3,4,5,7,8]]
    arrivals=[2.1,2.6,3.2,5.4,6.0,8.4,9.0]
    heights=[37]+[38]*6
    layer=Canvas()
    layer.rrect(891,695,938,261,0,fill=PAPER)
    y=703
    # Once the final row has settled, every displayed quote holds through station exit.
    offset=41*ease(prog(t,8.35,8.85))
    for i,(e,at,h) in enumerate(zip(rows,arrivals,heights)):
        a=ease(prog(t,at,at+.3))
        if a<=0: continue
        yy=y-offset+(1-a)*8
        if e['who']=='notify':
            pill(layer,914,yy+12,f"caregiver alerted · {e['level']}",ALERT_INK,ALERT_BG,12)
        else:
            wrong=i==6
            if wrong:
                card(layer,901,yy-3,915,60,ALERT,ALERT_BG,1.8)
            layer.text(914,yy,f"t={e['t']:g} s",mono(13),mix(FAINT,PAPER,a))
            layer.text(1026,yy,'Jean' if e['who']=='person' else 'Agent',sans(15,'SemiBold'),mix(GRAY,PAPER,a))
            txt(layer,1092,yy,clean(e['text']),sans(19),704,25,mix(INK,ALERT_BG if wrong else PAPER,a))
            if i==5:
                pill(layer,1578,yy+12,pretty(e['state']),WARM_INK,WARM_BG,11)
            if wrong:
                layer.text(1092,yy+32,'wrong for someone on the floor',mono(12,'Medium'),ALERT_INK)
        y+=h
    box=(sc(891),sc(695),sc(1829),sc(956))
    tile=layer.im.crop(box)
    # Paper backing for the clipped layer.
    # Canvas's background is warm ground; use only the actual viewport, inside the border.
    mask=Image.new('L',tile.size,0)
    ImageDraw.Draw(mask).rounded_rectangle((0,0,tile.width-1,tile.height-1),sc(10),fill=255)
    cv.im.paste(tile,(box[0],box[1]),mask)


def checks(cv,t):
    card(cv,890,200,940,190)
    cv.text(915,222,'AUTOMATIC TRACE CHECKS',mono(13,'Medium'),FAINT)
    for row,prefix in enumerate(['TT','SM','TM']):
        ids=[s for s in F['invariants'] if s.startswith(prefix)]
        for j,s in enumerate(ids):
            pill(cv,917+j*112,267+row*39,s,INK,PAPER,14,a=ease(prog(t,.2+j*.04+row*.2,.5+j*.04+row*.2)))
    cv.text(915,406,'TT turn-taking · SM state machine · TM timing',sans(17),GRAY)
    card(cv,890,451,940,288)
    cv.text(915,473,F['scene']['id'],mono(13),FAINT)
    left,right,axis=920,1800,562
    scale=lambda s:left+(right-left)*s/300
    cv.rrect(scale(F['transcript'][0]['t']),axis-25,right-scale(F['transcript'][0]['t']),50,4,fill=WARM_BG)
    cv.text(1360,529,pretty(F['transcript'][0]['state']),mono(12),WARM_INK,'ma')
    cv.polyline([(left,axis),(right,axis)],FAINT,1.6)
    for e in F['transcript']:
        if e['who']=='notify' or e['t']>300: continue
        x=scale(e['t']); dy=-13 if e['who']=='person' else 13
        cv.polyline([(x,axis),(x,axis+dy)],GRAY,1.4)
    cv.text(left,608,'Jean above · Agent below',mono(12),FAINT)
    cv.text(right,608,'scene time · seconds',mono(12),FAINT,'ra')
    # Axis scale is layout, not a captured measurement.
    for tick in [0,60,120,180,240,300]:
        cv.text(scale(tick),axis+18,str(tick),mono(11),FAINT,'ma')
    for i,f in enumerate(F['flags']):
        a=ease(prog(t,1.4+i*.6,1.9+i*.6))
        if not a: continue
        x=scale(f['t']); y=644+i*23
        col=ALERT if f['severity']=='major' else WARM
        cv.polyline([(x,axis-19),(x,600)],mix(col,PAPER,a),1)
        circle(cv,x,axis-19,4,mix(col,PAPER,a))
        evidence=f['evidence'][0].split(': ',1)[1]
        label=f"{f['check']} {pretty(f['summary'])} · {f['severity']} · t={f['t']:g} s"
        cv.text(925,y,label,mono(13),mix(col,PAPER,a))
        cv.text(1375,y,f'“{evidence}”',sans(17),mix(INK,PAPER,a))
    card(cv,890,775,940,145)
    a=ease(prog(t,3.7,5.3))
    cv.text(916,794,str(round(F['run_bug_entries']*a)),serif(64),INK)
    cv.text(1026,825,f"entries in bugs.jsonl from {round(F['run_scenes']*a)} scenes",sans(24),INK)
    cv.text(900,947,'the checks read the trace; no scene needs a hand-written answer key',mono(13),FAINT)


def triage(cv,t):
    labels=[('bugs.jsonl + reports + code','THE EVIDENCE'),(F['roles']['triage']+' · reads, traces the cause','TRIAGE'),('throwaway git worktree · patch + replay','TRY IT')]
    for i,(label,kicker) in enumerate(labels):
        x=890+i*326
        card(cv,x,210,288,125)
        cv.text(x+18,231,kicker,mono(12,'Medium'),FAINT)
        txt(cv,x+18,264,label,sans(20,'SemiBold'),250,27,t=t,start=i*.25,duration=.6)
        if i<2: cv.wire([(x+295,273),(x+320,273)],FAINT,1.5)
    card(cv,890,370,940,535)
    cv.text(915,391,f"fixes.md  ·  item {F['fix_items'][1]['n']} of {len(F['fix_items'])}",mono(13,'Medium'),FAINT)
    txt(cv,915,426,clean(F['fix2']['title']),sans(24,'SemiBold'),875,31)
    y=510
    cv.text(915,y,'Cause',mono(13,'Medium'),FAINT)
    y+=29
    for i,line in enumerate(F['fix2']['cause'].splitlines()):
        line=clean(line.strip().removeprefix('- '))
        loc,body=line.split(' ',1)
        cv.text(915,y,words(loc,t,1.3+i*.65,.15),mono(15),INK)
        y+=txt(cv,1055,y,body,sans(17),735,24,GRAY,t=t,start=1.4+i*.65,duration=.55)+12
    # Fixed section positions prevent typing from moving the following content.
    cv.text(915,668,'Proposal',mono(13,'Medium'),FAINT)
    proposal=clean(F['fix2']['proposal'].splitlines()[0].removeprefix('- '))
    txt(cv,915,697,proposal,sans(17),878,25,INK,t=t,start=3.0,duration=1.0)
    cv.text(915,770,'Quick test',mono(13,'Medium'),FAINT)
    quick=clean(F['fix2']['quick_test'])
    match=re.search(r'Replayed .*?\(it was sent twice\)\.',quick)
    txt(cv,915,799,match.group(),sans(18),874,27,INK,t=t,start=4.4,duration=1)
    if t>5.4:
        value=re.search(r'(\d+ times)',quick).group()
        pill(cv,915,867,value,COOL,COOL_BG,13,a=ease(prog(t,5.4,5.8)))
    cv.text(900,944,'no Docker, no network, no commits; only a copy to try the fix in',mono(13),FAINT)


def owner(cv,t):
    for i,item in enumerate(F['fix_items']):
        y=206+i*55
        a=ease(prog(t,i*.1,.35+i*.1))
        if not a: continue
        selected=item['n']==F['fix_items'][1]['n']
        card(cv,890,y,940,47,mix(INK if selected else LINE_SOFT,GROUND,a),mix(PAPER,GROUND,a),2 if selected else 1)
        cv.text(907,y+13,str(item['n']),mono(14),mix(FAINT,PAPER,a))
        title=ellipsize(clean(item['title']),sans(17),644)
        cv.text(950,y+12,title,sans(17),mix(INK,PAPER,a))
        approve=selected and t>=2.3
        label=f"   applied in {F['commit']['sha']}" if approve else ('asks the owner' if item['owner']!='none' else 'no decision needed')
        f=COOL if approve else (WARM_INK if item['owner']!='none' else GRAY)
        bg=COOL_BG if approve else (WARM_BG if item['owner']!='none' else PAPER)
        pw=mono(11,'Medium').getlength(label)/S+22
        pill(cv,1810-pw,y+24,label,f,bg,11,a=a*ease(prog(t,2.3,2.7)) if approve else a)
        if approve:
            aa=ease(prog(t,2.3,2.7))
            cv.polyline([(1824-pw,y+24),(1827-pw,y+27),(1833-pw,y+20)],mix(COOL,COOL_BG,aa),1.6,frac=aa)
    txt(cv,910,856,'Nothing is applied until the owner asks for it. Open policy questions go to them, not into code.',sans(20),884,30)


def correct(cv,t):
    g=F['guidelines']
    card(cv,890,200,940,370)
    cv.text(916,222,'WHAT COUNTS AS CORRECT  ·  PUBLISHED GUIDANCE',mono(13,'Medium'),FAINT)
    txt(cv,916,262,f"A guideline pack of {g['clauses']} clauses, each a paraphrase of one source passage with a link.",
        sans(22),880,32,t=t,start=.2,duration=1.2)
    x,y=916,345
    for i,name in enumerate(g['sources']):
        a=ease(prog(t,.9+i*.15,1.2+i*.15))
        f=mono(13,'Medium'); w=f.getlength(name)/S+24
        if x+w>1806: x,y=916,y+40
        pill(cv,x,y+12,name,INK,PAPER,13,a=a)
        x+=w+10
    a=ease(prog(t,2.2,2.6))
    cv.text(916,446,f"{g['checked']} of {g['clauses']} checked against the source by a person",sans(20,'SemiBold'),mix(INK,PAPER,a))
    for i in range(g['clauses']):
        aa=ease(prog(t,2.4+i*.05,2.6+i*.05))
        ok=i<g['checked']
        bx=916+i*34
        cv.rrect(bx,490,24,24,5,fill=mix(COOL if ok else PAPER,PAPER,aa),outline=mix(COOL if ok else LINE,PAPER,aa),width=1.4)
    card(cv,890,595,940,300)
    cv.text(916,617,'AND PEOPLE',mono(13,'Medium'),FAINT)
    rows=[("Benchmark labels","drafted by an isolated Claude that never sees the agent's code; a human reviews every one"),
          ("What Jean hears","the caregiver writes the phrases"),
          ("Timings","where the guidance is silent, a caregiver setting, never cited as evidence"),
          ("Open questions","answered by the owner after each night")]
    yy=656
    for i,(k,v) in enumerate(rows):
        a=ease(prog(t,3.4+i*.6,3.8+i*.6))
        lines=wrap(v,sans(19),620)[:2]
        cv.text(916,yy,k,sans(19,'SemiBold'),mix(INK,PAPER,a))
        for j,ln in enumerate(lines):
            cv.text(1110,yy+j*25,ln,sans(19),mix(GRAY,PAPER,a))
        yy+=25*len(lines)+22
    cv.text(910,918,'not clinical validation: every rule traces to a source or a named human decision',mono(13),FAINT)


def landed(cv,t):
    commit=F['commit']
    card(cv,890,200,940,340)
    cv.text(915,220,f"{commit['sha']}  ·  {commit['date']}",mono(15),FAINT)
    txt(cv,915,254,commit['subject'],sans(20),886,28)
    card(cv,907,308,906,209,LINE_SOFT,(239,238,232))
    snippet=textwrap.dedent('\n'.join(commit['veto_snippet'])).splitlines()
    # Long captured code is wrapped at token boundaries, never rewritten.
    for i,line in enumerate(snippet):
        y=326+i*24
        a=ease(prog(t,.4+i*.23,.65+i*.23))
        if not a: continue
        for ln in ([line] if mono(16).getlength(line)/S<=868 else wrap(line,mono(16),868)):
            cv.text(925,y,ln,mono(16),mix(INK,(239,238,232),a)); y+=24
    for j,key in enumerate(['before','after']):
        d=F[key]; x=890+j*484; col=ALERT_INK if j==0 else COOL
        card(cv,x,575,456,365)
        cv.text(x+24,599,'Before the fix' if j==0 else 'After the fix',sans(22,'SemiBold'),INK)
        cv.text(x+24,642,str(d['prompts']),serif(72),col)
        txt(cv,x+24,731,'times told to get back to bed while on the floor',sans(19),402,27)
        small=(f"{d['scenes_affected']} of {d['floor_scenes']} floor scenes · {len(d['runs'])} runs" if j==0 else f"{d['floor_scenes']} floor scenes · {len(d['runs'])} runs")
        cv.text(x+24,819,small,mono(13),GRAY)
        for i in range(d['floor_scenes']):
            a=ease(prog(t,2+i*.08,2.25+i*.08))
            fill=ALERT if i<d['scenes_affected'] else LINE
            circle(cv,x+27+(i%18)*22,885,5,mix(fill,PAPER,a))


def outro(cv,t):
    card(cv,890,325,940,335)
    cv.text(916,353,'THE LOOP, ACROSS REAL RUNS',mono(13,'Medium'),FAINT)
    z=F['totals']
    txt(cv,916,408,f"{z['director_hours']} director hours · {z['scenes']} simulated scenes · {z['bug_entries']} flags · {z['fix_items']} ranked fixes",sans(26),860,38)
    txt(cv,916,548,'Claude finds and proposes. The agent stays local. A person decides.',serif_i(24),850,34)


PANELS=[intro,simulate,checks,triage,owner,correct,landed,outro]


def panel_image(index,local):
    cv=Canvas()
    PANELS[index](cv,local)
    return cv.im.crop((sc(870),sc(190),sc(1850),sc(980)))


def frame(t):
    cv=Canvas()
    loop_map(cv,t)
    cv.text(77,58,f"NIGHT COMPANION  ·  SCENE_LAB  ·  run {F['run']}",mono(15,'Medium'),FAINT)
    index=max(i for i,s in enumerate(STARTS) if t>=s)
    a=ease(prog(t,STARTS[index],STARTS[index]+.5)) if index else 1
    # Panels crossfade within a bounded viewport, drifting upward on entry.
    current=panel_image(index,t-STARTS[index])
    if index and a<1:
        prior=panel_image(index-1,STARTS[index]-STARTS[index-1])
        current=Image.blend(prior,current,a)
    cv.im.paste(current,(sc(870),sc(190+(1-a)*12)))
    if index and a<1:
        cv.text(77,84,HEADINGS[index-1],serif(46),mix(INK,GROUND,1-a))
    cv.text(77,84,HEADINGS[index],serif(46),mix(INK,GROUND,a))
    footer=f"Real run {F['run']} · Claude via the claude.ai subscription · the bedside agent runs {F['agent_model']} locally"
    cv.text(77,1030,footer,mono(13),FAINT)
    cv.text(1843,1030,'not a medical device',mono(13),FAINT,'ra')
    return fade_all(cv.final(),t,DUR)


def init(path):
    global F
    F=json.loads(Path(path).read_text())


def render_one(k):
    return frame(k/FPS).tobytes()


def main():
    from motion import WORK
    path=Path(sys.argv[1]) if len(sys.argv)>1 else WORK/'scene-lab-loop-2026-10-01'/'loop_facts.json'
    out=Path(sys.argv[2]) if len(sys.argv)>2 else WORK/'scene-lab-loop-2026-10-01'/'scene-lab-loop-30fps.mp4'
    out.parent.mkdir(parents=True,exist_ok=True)
    init(path)
    if out.suffix.lower()=='.png':
        t=float(sys.argv[3]) if len(sys.argv)>3 else 3
        frame(t).save(out)
        print('wrote',out)
    else:
        encode(render_one,round(DUR*FPS),out,init=init,initargs=(str(path),))


if __name__=='__main__':
    main()
