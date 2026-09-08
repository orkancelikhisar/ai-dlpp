#!/usr/bin/env python3
"""Every figure in the paper, as SVG, from docs/paper/data/numbers.json. Pure Python.

Colour code, used everywhere: grey = in-browser 2-4B arms; orange = trivial floors and
oracles; blue = hosted 30-120B arms, thinking OFF; purple = thinking ON.
"""
import json, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "data/numbers.json")))
OUT = os.path.join(HERE, "figures")
C = {"local": "#8d97a3", "floor": "#e0782f", "ceil": "#2b6cb0", "ceilB": "#7fb0e0", "think": "#6b3fa0", "ink": "#1f2937", "mute": "#6b7280", "grid": "#e5e7eb", "bg": "#ffffff"}
FONT = "font-family='Helvetica Neue, Helvetica, Arial, sans-serif'"

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;")
def text(x, y, s, size=10, anchor="start", color=C["ink"], weight="normal", rot=None, style=""):
    t = f"transform='rotate({rot} {x} {y})'" if rot is not None else ""
    return f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' text-anchor='{anchor}' fill='{color}' font-weight='{weight}' {t} style='{style}'>{esc(s)}</text>"
def rect(x, y, w, h, fill, stroke="none", rx=1.5, extra=""):
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w,0):.1f}' height='{max(h,0):.1f}' fill='{fill}' stroke='{stroke}' rx='{rx}' {extra}/>"
def line(x1, y1, x2, y2, color=C["grid"], w=1, dash=None):
    d = f"stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{w}' {d}/>"
def circ(x, y, r, fill, stroke="white"): return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='1'/>"
def svg(w, h, body, name):
    doc = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img'>" + rect(0,0,w,h,C["bg"],rx=0) + "".join(body) + "</svg>"
    open(os.path.join(OUT, name), "w", encoding="utf8").write(doc); print("  wrote", name)

def yaxis(x0, y0, y1, vmax, ticks, label=None, w=None):
    b = []
    for t in ticks:
        y = y1 - (y1-y0) * t / vmax
        b.append(line(x0, y, (w or x0+6), y)); b.append(text(x0-4, y+3.5, f"{t:g}", 9, "end", C["mute"]))
    if label: b.append(text(x0-30, (y0+y1)/2, label, 9.5, "middle", C["mute"], rot=-90))
    return b

# ---------------------------------------------------------------- F2 headline
def f2():
    W, H = 720, 250; b = []
    ml = N["message_level"]; sw = N["spanwise_predicate"]; sl = N["spanlevel_entity"]
    jd = "judge-deepseek-v4-flash-0731"
    jd_msg = [ml[jd][f"ceiling-0{p}"]["F1"] for p in (1,2,3)]
    jd_sw = [float(sw[f"ceiling-judge-deepseek-v4-flash-0731 [ceiling-0{p}]"]["F1"]) for p in (1,2,3)]
    nem = [float(sl[f"ceiling-b-nemotron-3-super-120b-a12b [ceiling-0{p}]"]["F1"]) for p in (1,2,3)]
    local_msg_best = max(v["F1"] for v in N["local_message_level"].values())
    panels = [
        ("A  Predicate, span-wise (as published)", [("in-browser\nbest (2–4 B)", 0.197, C["local"], None), ("floor\n(capitalised run)", 0.571, C["floor"], None), ("hosted best\n(DeepSeek judge)", sum(jd_sw)/3, C["ceil"], jd_sw)]),
        ("B  Predicate, message level (gold's unit)", [("in-browser\nbest (2–4 B)", local_msg_best, C["local"], None), ("floor\n(capitalised run)", N["floor_message_level_all"]["F1"], C["floor"], None), ("hosted best\n(DeepSeek judge)", sum(jd_msg)/3, C["ceil"], jd_msg)]),
        ("C  Entity spans (108 gold)", [("in-browser\nbest", 0.331, C["local"], None), ("oracle\nno labels", float(N["span_floors"]["unbudgeted"]["F1"]), C["floor"], None), ("oracle\ntold N", float(N["span_floors"]["budget-matched"]["F1"]), C["floor"], None), ("hosted best\nthink OFF", sum(nem)/3, C["ceil"], nem), ("GLM B\nthink ON", float(sl["ceiling-b-glm-5.3-flash [thinkonglm-01] **thinking ON**"]["F1"]), C["think"], None)]),
    ]
    widths = [200, 200, 282]; gap = 12; y0, y1 = 34, 196; xs = [10, 10+widths[0]+gap, 10+widths[0]+widths[1]+2*gap]
    for i, (title, bars) in enumerate(panels):
        pw = widths[i]; x0 = xs[i]; ax = x0 + 30
        b.append(text(x0, 16, title, 10.5, "start", C["ink"], "bold"))
        b += yaxis(ax, y0, y1, 1.0, [0, .25, .5, .75, 1.0], "F1", w=x0+pw)
        b.append(line(ax, y1, x0+pw, y1, C["mute"]))
        n = len(bars); bw = (pw-34) / n * 0.62; step = (pw-34)/n
        for k, (lab, v, col, dots) in enumerate(bars):
            cx = ax + 4 + step*k + step/2; y = y1 - (y1-y0)*v
            hatch = col == C["floor"]
            b.append(rect(cx-bw/2, y, bw, y1-y, col if not hatch else "#f6c9a6", stroke=C["floor"] if hatch else "none"))
            if dots:
                for d in dots: b.append(circ(cx, y1-(y1-y0)*d, 3, "white", col))
            b.append(text(cx, y-4, f"{v:.3f}", 9.5, "middle", C["ink"], "bold"))
            for j, ln in enumerate(lab.split("\n")): b.append(text(cx, y1+11+j*10, ln, 8.5, "middle", C["mute"]))
    b.append(text(10, H-6, "Bars: F1 over all 179 scored items (panels A, B) or all 189 items (C). Hollow dots: the three passes at temperature 0; bar = mean. Orange = a reader with no model.", 8.5, "start", C["mute"]))
    svg(W, H, b, "f2-headline.svg")

# ---------------------------------------------------------------- F3 per-pass + variance
def f3():
    W, H = 700, 312; b = []
    ml = N["message_level"]; sp = N["variance"]["per_arm"]
    arms = sorted(sp, key=lambda a: -sum(sp[a]["passes"])/3)
    x0, y0, y1 = 52, 30, 262; pw = 470
    b.append(text(10, 16, "A  Message-level predicate F1, every hosted arm, three passes at temperature 0", 10.5, "start", C["ink"], "bold"))
    b += yaxis(x0, y0, y1, 1.0, [0,.25,.5,.75,1.0], "F1", w=x0+pw)
    fl = N["floor_message_level_all"]["F1"]; yf = y1-(y1-y0)*fl
    b.append(line(x0, yf, x0+pw, yf, C["floor"], 1.5, "5,3")); b.append(text(x0+pw-2, yf+11, f"floor {fl:.3f}", 9, "end", C["floor"], "bold"))
    step = pw/len(arms)
    for k, a in enumerate(arms):
        cx = x0 + step*k + step/2; v = sp[a]["passes"]; col = C["ceil"] if a.startswith("judge") else C["ceilB"]
        ylo, yhi = y1-(y1-y0)*min(v), y1-(y1-y0)*max(v)
        b.append(line(cx, ylo, cx, yhi, col, 2))
        for p, d in enumerate(v): b.append(circ(cx, y1-(y1-y0)*d, 3.2, col if p<2 else "white", col if p==2 else "white"))
        b.append(text(cx, yhi-7, f"{sp[a]['spread']:.2f}" if sp[a]["spread"]>0.05 else "", 8, "middle", C["mute"]))
        lab = a.replace("-v4-flash-0731","").replace("-3-super-120b-a12b","").replace("-small-2603","").replace("qwen3.8-","qwen-").replace("judge-","J·").replace("b-","B·")
        b.append(text(cx+3, y1+9, lab, 8.5, "end", C["mute"], rot=-40))
    # panel B: spread comparison
    bx = 560; b.append(text(bx-8, 16, "B  Pass-to-pass spread", 10.5, "start", C["ink"], "bold"))
    b += yaxis(bx+20, y0, y1, 0.4, [0,.1,.2,.3,.4], "F1 spread, mean (dot = max)", w=bx+130)
    span_mean, span_max = N["variance"]["entity_mean_spread"], N["variance"]["entity_max_spread"]
    for k, (lab, m, mx, col) in enumerate([("entity\nspans", span_mean, span_max, C["ceilB"]), ("predicate,\nmessage-level", N["variance"]["mean_spread"], N["variance"]["max_spread"], C["ceil"])]):
        cx = bx + 45 + k*55; y = y1-(y1-y0)*m/0.4
        b.append(rect(cx-16, y, 32, y1-y, col)); b.append(circ(cx, y1-(y1-y0)*mx/0.4, 3.5, col))
        b.append(text(cx, y-5, f"{m:.3f}", 9.5, "middle", C["ink"], "bold"))
        for j, ln in enumerate(lab.split("\n")): b.append(text(cx, y1+11+j*10, ln, 8.5, "middle", C["mute"]))
    b.append(text(10, H-6, "Dark dots = passes 1–2, hollow = pass 3; number above = spread (max − min) where it exceeds 0.05. J = compiled judge, B = policy-in-context.", 8.5, "start", C["mute"]))
    svg(W, H, b, "f3-variance.svg")

# ---------------------------------------------------------------- F5 thinking-ON
def f5():
    W, H = 720, 256; b = []
    t6, t8 = N["thinkon"]["glm_600"], N["thinkon"]["glm_8192"]
    b.append(text(10, 16, "A  GLM judge, thinking ON: 600 vs 8,192-token cap (message level, answered rows)", 10.5, "start", C["ink"], "bold"))
    x0, y0, y1 = 48, 32, 190; pw = 330
    b += yaxis(x0, y0, y1, 1.0, [0,.25,.5,.75,1.0], "", w=x0+pw); b.append(line(x0,y1,x0+pw,y1,C["mute"]))
    groups = [("precision", "P"), ("recall", "R"), ("F1", "F1")]
    gs = pw/3
    for g, (lab, key) in enumerate(groups):
        for k, (blk, col, name) in enumerate([(t6, "#b39ddb", "600"), (t8, C["think"], "8,192")]):
            v = blk["message_attempted"][key]; cx = x0 + gs*g + gs*(0.3 + 0.4*k); bw = gs*0.3
            y = y1-(y1-y0)*v; b.append(rect(cx-bw/2, y, bw, y1-y, col)); b.append(text(cx, y+11, f"{v:.3f}", 8.5, "middle", "white", "bold"))
            if g == 0: b.append(text(cx, y1+22, name, 8.5, "middle", C["mute"]))
        b.append(text(x0+gs*g+gs/2, y1+11, lab, 9.5, "middle", C["ink"]))
        if key == "F1":
            for blk, col in [(t6,"#b39ddb"),(t8,C["think"])]:
                fl = blk["floor_attempted"]["F1"]; yf = y1-(y1-y0)*fl
                b.append(line(x0+gs*g+4, yf, x0+gs*g+gs-4, yf, C["floor"], 1.5, "4,3"))
            pass
    # panel B stacked split
    bx = 440; b.append(text(bx, 16, "B  The gold positives, by call outcome", 10.5, "start", C["ink"], "bold"))
    cats = [("detected_clean","detected, call finished", C["think"]), ("detected_truncated","detected, call truncated", "#b39ddb"), ("missed_truncated","MISSED, call truncated", C["floor"]), ("missed_clean","missed, call finished", "#c9c9c9")]
    for k, (blk, name) in enumerate([(t6,"600 cap"),(t8,"8,192 cap")]):
        cx = bx + 40 + k*88; tot = sum(blk["split"].values()); y = y1; bw = 50
        for key, lab, col in cats:
            n = blk["split"][key]; h = (y1-y0)*n/19
            if n: b.append(rect(cx-bw/2, y-h, bw, h, col)); b.append(text(cx, y-h/2+3.5, str(n), 9.5, "middle", "white" if col!="#c9c9c9" else C["ink"], "bold")); y -= h
        b.append(text(cx, y1+11, name, 9.5, "middle", C["ink"])); b.append(text(cx, y1+22, f"{blk['positives_answered']} of 19 answered", 8.5, "middle", C["mute"]))
    for j, (key, lab, col) in enumerate(cats):
        b.append(rect(bx+172, 40+j*14, 9, 9, col)); b.append(text(bx+185, 48+j*14, lab, 8, "start", C["mute"]))
    b.append(text(bx, 228, f"truncated rows {t6['truncated_rows']} → {t8['truncated_rows']} of 189;  unanswered (429s) {189-t6['answered']} → {189-t8['answered']}", 8.5, "start", C["mute"]))
    b.append(text(10, H-6, f"Dashed: floors recomputed over the same answered rows ({t6['floor_attempted']['F1']:.3f} at 600, {t8['floor_attempted']['F1']:.3f} at 8,192). Every miss at 600 was a truncated call, with no clean misses; at 8,192 the missed list is empty.", 8.5, "start", C["mute"]))
    svg(W, H, b, "f5-thinking-on.svg")

# ---------------------------------------------------------------- F6 latency
def f6():
    W, H = 700, 250; b = []
    L = N["latency"]; models = ["mistral-small-2603", "qwen3.8-27b", "qwen3.8-flash", "deepseek-v4-flash-0731", "nemotron-3-super-120b-a12b"]
    short = {"mistral-small-2603":"Mistral Small 24B","qwen3.8-27b":"Qwen3.8 27B","qwen3.8-flash":"Qwen3.8 Flash","deepseek-v4-flash-0731":"DeepSeek V4 Flash","nemotron-3-super-120b-a12b":"Nemotron 3 Super 120B"}
    b.append(text(10, 16, "Per-call wall time p50 (ms), pooled over three passes — compiled judge vs policy-in-context on the same model", 10.5, "start", C["ink"], "bold"))
    x0, xw = 150, 440; vmax = 5000; y = 34; rh = 17
    for t in [0,1000,2000,3000,4000,5000]:
        x = x0 + xw*t/vmax; b.append(line(x, 32, x, 226)); b.append(text(x, 29, f"{t:,} ms", 8, "middle", C["mute"]))
    for m in models:
        b.append(text(x0-6, y+rh-2, short[m], 9.5, "end", C["ink"]))
        for k, (fam, col) in enumerate([("judge", C["ceil"]), ("b", C["ceilB"])]):
            a = L[f"{fam}-{m}"]; yy = y + k*rh
            tt = x0 + xw*a["ttft_p50"]/vmax; cw = x0 + xw*a["call_wall_p50"]/vmax
            b.append(rect(x0, yy+2, cw-x0, rh-5, col)); b.append(rect(x0, yy+2, tt-x0, rh-5, "#1e3a5f" if fam=="judge" else "#4c7fb4"))
            b.append(text(cw+4, yy+rh-3, f"{a['call_wall_p50']:,} ms · {a['completion_p50']} tok" + (f" · {a['n429']} ×429" if a["n429"] else ""), 8.5, "start", C["mute"]))
        y += rh*2 + 8
    b.append(rect(x0, 240, 9, 9, "#1e3a5f")); b.append(text(x0+13, 248, "time to first token", 8.5)); b.append(rect(x0+120, 240, 9, 9, C["ceil"])); b.append(text(x0+133, 248, "judge, decode", 8.5))
    b.append(rect(x0+220, 240, 9, 9, "#4c7fb4")); b.append(text(x0+233, 248, "B, first token", 8.5)); b.append(rect(x0+320, 240, 9, 9, C["ceilB"])); b.append(text(x0+333, 248, "B, decode", 8.5))
    b.append(text(x0+420, 248, "tok = completion tokens p50", 8.5, "start", C["mute"]))
    svg(W, H, b, "f6-latency.svg")

# ---------------------------------------------------------------- F8 P/R scatter
def f8(W=340, H=316, y1=272, name="f8-pr-scatter.svg", compact=False):
    b = []
    x0, y0, x1 = 40, 40, 325
    def X(p): return x0 + (x1-x0)*p
    def Y(r): return y1 - (y1-y0)*r
    b.append(text(10, 14, "Message-level predicate: precision vs recall, every arm", 10.5, "start", C["ink"], "bold"))
    for t in [0,.25,.5,.75,1.0]:
        b.append(line(X(t), y0, X(t), y1)); b.append(text(X(t), y1+11, f"{t:g}", 8.5, "middle", C["mute"]))
        b.append(line(x0, Y(t), x1, Y(t))); b.append(text(x0-4, Y(t)+3, f"{t:g}", 8.5, "end", C["mute"]))
    b.append(text((x0+x1)/2, y1+23, "precision", 9.5, "middle", C["mute"])); b.append(text(x0-28, (y0+y1)/2, "recall", 9.5, "middle", C["mute"], rot=-90))
    for f in [0.3,0.5,0.7,0.9]:  # iso-F1
        pts = []
        for i in range(1,100):
            p = i/100; r = f*p/(2*p-f) if 2*p-f>0 else None
            if r and 0<r<=1: pts.append(f"{X(p):.1f},{Y(r):.1f}")
        if pts: b.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='#d1d5db' stroke-width='0.8' stroke-dasharray='2,2'/>")
        b.append(text(X(min(1, f/(2-f))+0.02) if f<0.9 else x1-2, Y(f)-2 if f<0.9 else Y(0.9)+10, f"F1={f}", 7.5, "end", "#9ca3af"))
    for k, v in N["local_message_level"].items(): b.append(circ(X(v["P"]), Y(v["R"]), 3, C["local"], "none"))
    for arm, byrun in N["message_level"].items():
        for run, v in byrun.items():
            if run.startswith("ceiling-0"): b.append(circ(X(v["P"]), Y(v["R"]), 3.4, C["ceil"] if arm.startswith("judge") else C["ceilB"]))
    fl = N["floor_message_level_all"]; b.append(f"<rect x='{X(fl['P'])-4.5:.1f}' y='{Y(fl['R'])-4.5:.1f}' width='9' height='9' fill='{C['floor']}' stroke='white'/>")
    b.append(text(X(fl["P"])-7, Y(fl["R"])+4, "floor", 8.5, "end", C["floor"], "bold"))
    t8 = N["thinkon"]["glm_8192"]["message_attempted"]; b.append(f"<polygon points='{X(t8['P']):.1f},{Y(t8['R'])-6:.1f} {X(t8['P'])+5.5:.1f},{Y(t8['R'])+4:.1f} {X(t8['P'])-5.5:.1f},{Y(t8['R'])+4:.1f}' fill='{C['think']}' stroke='white'/>")
    b.append(text(X(t8["P"])-8, Y(t8["R"])+34, "GLM think-ON (8,192)", 8, "end", C["think"], "bold"))
    jd = N["message_level"]["judge-deepseek-v4-flash-0731"]["ceiling-03"]; b.append(text(X(jd["P"])-8, Y(jd["R"])+22, "DeepSeek judge ×3", 8, "end", C["ceil"], "bold"))
    b.append(text(X(0.05), Y(0.62), "in-browser 2–4 B: low precision at any recall", 8, "start", C["local"]))
    if not compact: b.append(text(10, H-6, f"Grey: {len(N['local_message_level'])} in-browser arms on the v2 corpus. Blue: hosted judge (dark) and B (light), 3 passes each. Dashed: iso-F1.", 8.5, "start", C["mute"]))
    svg(W, H, b, name)

# ---------------------------------------------------------------- F7 cost
def f7():
    W, H = 340, 230; b = []
    S = N["spend"]; bym = S["byModel"]
    b.append(text(10, 14, "Billed cost per model, both families, all runs (USD)", 10.5, "start", C["ink"], "bold"))
    names = list(bym); vmax = 0.4; x0, xw = 150, 172; y = 30; rh = 18
    for t in [0,.1,.2,.3,.4]:
        x = x0 + xw*t/vmax; b.append(line(x, 24, x, 24+rh*len(names)+4)); b.append(text(x, 24+rh*len(names)+16, f"${t:.1f}", 8.5, "middle", C["mute"]))
    short = {"qwen3.8-27b":"Qwen3.8 27B","nemotron-3-super-120b-a12b":"Nemotron 3 Super","glm-5.3-flash":"GLM-5.3 Flash (think ON only)","qwen3.8-flash":"Qwen3.8 Flash","mistral-small-2603":"Mistral Small","deepseek-v4-flash-0731":"DeepSeek V4 Flash"}
    for m in names:
        v = bym[m]; w = xw*v/vmax; col = C["think"] if "glm" in m else C["ceil"]
        b.append(text(x0-6, y+13, short.get(m,m), 9.5, "end", C["ink"])); b.append(rect(x0, y+3, w, rh-6, col)); b.append(text(x0+w+4, y+13, f"${v:.3f}", 8.5, "start", C["mute"])); y += rh
    y += 26
    b.append(text(10, y, f"Total billed ${S['costUsd']:.3f} over {S['calls']:,} calls, of a $10 key ({100*S['costUsd']/10:.0f}%).", 9.5, "start", C["ink"], "bold")); y += 13
    T = S["thinkoff"]
    b.append(text(10, y, f"Thinking OFF only: judge ${T['byFamily']['judge']:.3f} vs B ${T['byFamily']['b']:.3f} — B costs {T['byFamily']['b']/T['byFamily']['judge']:.2f}× (prompt size).", 9, "start", C["mute"])); y += 12
    b.append(text(10, y, f"Price-table estimate ${T['estimateUsd']:.3f} vs ${T['costUsd']:.3f} billed: {100*(T['estimateUsd']/T['costUsd']-1):.0f}% over.", 9, "start", C["mute"]))
    svg(W, H, b, "f7-cost.svg")

# ---------------------------------------------------------------- F1 architecture
def f1():
    W, H = 700, 200; b = []
    def box(x, y, w, h, title, sub, fill="#f3f4f6", stroke="#9ca3af"):
        b.append(rect(x, y, w, h, fill, stroke, 4)); b.append(text(x+w/2, y+15, title, 9.5, "middle", C["ink"], "bold"))
        for j, s in enumerate(sub): b.append(text(x+w/2, y+28+j*11, s, 8.3, "middle", C["mute"]))
    def arrow(x1, y1, x2, y2, col="#6b7280"):
        b.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='1.4' marker-end='url(#ah)'/>")
    b.append("<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#6b7280'/></marker></defs>")
    b.append(text(10, 14, "Compile once in the cloud; detect every prompt locally", 10.5, "start", C["ink"], "bold"))
    box(10, 26, 118, 60, "Policy document", ["natural language,", "provider clauses", "(p-fin: 5,320 bytes)"])
    box(150, 26, 118, 60, "Compiler", ["frontier model,", "compile-time only —", "touches no user data"], "#fff7ed", C["floor"])
    box(290, 26, 118, 60, "Policy IR", ["8 entity types · 10 rules", "1 semantic predicate (message)", "actions per provider"])
    arrow(128, 56, 150, 56); arrow(268, 56, 290, 56)
    b.append(line(430, 20, 430, 190, "#d1d5db", 1, "4,3")); b.append(text(436, 20, "runtime — in the browser, per prompt", 8.5, "start", C["mute"]))
    box(440, 30, 78, 46, "Tier 0", ["regex + validators", "sync"], "#eef2ff", "#818cf8")
    box(526, 30, 78, 46, "Tier 1", ["span tagger", "ONNX"], "#eef2ff", "#818cf8")
    box(612, 30, 78, 46, "Tier 2", ["LLM judge", "WebLLM / WebGPU"], "#eef2ff", "#818cf8")
    arrow(408, 56, 440, 53); arrow(518, 53, 526, 53); arrow(604, 53, 612, 53)
    box(440, 96, 250, 48, "Cluster-strictest action", ["block > redact > pseudonymize > allow", "pseudonyms rehydrated in the reply"], "#ecfdf5", "#34d399")
    arrow(565, 76, 565, 96)
    b.append(rect(10, 108, 398, 80, "#fafafa", "#d1d5db", 4))
    b.append(text(18, 124, "What this paper measures — the tier-2 judge, two ways to build it, two hardware classes", 9.5, "start", C["ink"], "bold"))
    b.append(text(18, 140, "Compiled judge: one predicate from the IR, ~300-token prompt.   Policy-in-context (B): the whole policy in the prompt, 1,410 tokens.", 8.3, "start", C["mute"]))
    b.append(rect(18, 150, 9, 9, C["local"])); b.append(text(31, 158, "in-browser arms: Qwen3.5-2B, Ministral-3-3B, Qwen3-4B, Phi-4-mini · 4-bit · WebGPU · 8,192 ctx · shippable", 8.3))
    b.append(rect(18, 166, 9, 9, C["ceil"])); b.append(text(31, 174, "capability-ceiling arms: six open-weight hosted models (27–120 B where published), providers pinned · non-shippable, same method", 8.3))
    b.append(rect(18, 178, 9, 9, C["think"])); b.append(text(31, 186, "+ one ceiling model with mandatory reasoning, thinking ON at two token caps", 8.3))
    svg(W, H, b, "f1-architecture.svg")

def e_scatter(): f8(W=340, H=232, y1=196, name="e-scatter.svg", compact=True)

def e_variance():
    W, H = 340, 178; b = []
    b.append(text(10, 14, "Pass-to-pass spread at temperature 0", 10.5, "start", C["ink"], "bold"))
    x0, y0, y1 = 44, 30, 128
    b += yaxis(x0, y0, y1, 0.4, [0,.1,.2,.3,.4], "F1 spread", w=x0+150)
    for k, (lab, m, mx, col) in enumerate([("entity spans", N["variance"]["entity_mean_spread"], N["variance"]["entity_max_spread"], C["ceilB"]), ("predicate (message)", N["variance"]["mean_spread"], N["variance"]["max_spread"], C["ceil"])]):
        cx = x0 + 45 + k*80; y = y1-(y1-y0)*m/0.4
        b.append(rect(cx-22, y, 44, y1-y, col)); b.append(circ(cx, y1-(y1-y0)*mx/0.4, 3.5, col))
        b.append(text(cx, y-5, f"mean {m:.3f}", 9, "middle", C["ink"], "bold")); b.append(text(cx+8, y1-(y1-y0)*mx/0.4+3, f"max {mx:.3f}", 8, "start", C["mute"]))
        b.append(text(cx, y1+12, lab, 8.5, "middle", C["mute"]))
    v = N["variance"]["per_arm"]["b-qwen3.8-flash"]["passes"]
    b.append(text(x0+170, 52, "Same inputs, three passes:", 9, "start", C["ink"], "bold"))
    b.append(text(x0+170, 66, "B on Qwen3.8 Flash scored", 8.5, "start", C["mute"]))
    b.append(text(x0+170, 82, f"{v[0]:.3f}  ·  {v[1]:.3f}  ·  {v[2]:.3f}", 11, "start", C["ceil"], "bold"))
    b.append(text(x0+170, 100, "6× the spread of span extraction.", 8.5, "start", C["mute"]))
    b.append(text(x0+170, 113, "One pass is not a measurement.", 8.5, "start", C["mute"]))
    b.append(text(10, H-6, "Ten hosted arms, passes 1–3, providers pinned; the sampler is not the source.", 8.5, "start", C["mute"]))
    svg(W, H, b, "e-variance.svg")

def e_thinkon():
    W, H = 340, 178; b = []
    t6, t8 = N["thinkon"]["glm_600"], N["thinkon"]["glm_8192"]
    b.append(text(10, 14, "Reasoning given room to finish: GLM-5.3 Flash", 10.5, "start", C["ink"], "bold"))
    x0, y0, y1 = 44, 30, 126; pw = 280
    b += yaxis(x0, y0, y1, 1.0, [0,.5,1.0], "", w=x0+pw); b.append(line(x0,y1,x0+pw,y1,C["mute"]))
    gs = pw/2
    for g, (lab, key) in enumerate([("recall", "R"), ("F1 vs floor", "F1")]):
        for k, (blk, col, name) in enumerate([(t6, "#b39ddb", "600-token cap"), (t8, C["think"], "8,192 cap")]):
            v = blk["message_attempted"][key]; cx = x0 + gs*g + gs*(0.3+0.4*k); bw = gs*0.28
            y = y1-(y1-y0)*v; b.append(rect(cx-bw/2, y, bw, y1-y, col)); b.append(text(cx, y+11, f"{v:.3f}", 8.5, "middle", "white", "bold"))
            if key == "F1":
                fl = blk["floor_attempted"]["F1"]; yf = y1-(y1-y0)*fl
                b.append(line(cx-bw/2-3, yf, cx+bw/2+3, yf, C["floor"], 1.6, "4,2")); b.append(text(cx, yf-4, f"floor {fl:.3f}", 7.5, "middle", C["floor"]))
            if g == 0: b.append(text(cx, y1+21, name, 8, "middle", C["mute"]))
        b.append(text(x0+gs*g+gs/2, y1+11, lab, 9, "middle", C["ink"]))
    b.append(text(10, H-18, f"At 600 tokens every miss was a truncated call (8 of 8); at 8,192 the missed list is empty.", 8.5, "start", C["mute"]))
    b.append(text(10, H-6, "Message level, rows each run answered; floors recomputed on the same rows. One pass; 31% of rows lost to 429s.", 8.5, "start", C["mute"]))
    svg(W, H, b, "e-thinkon.svg")

def e_latency():
    W, H = 340, 178; b = []
    L = N["latency"]; models = ["mistral-small-2603", "qwen3.8-27b", "qwen3.8-flash", "deepseek-v4-flash-0731", "nemotron-3-super-120b-a12b"]
    short = {"mistral-small-2603":"Mistral Small","qwen3.8-27b":"Qwen3.8 27B","qwen3.8-flash":"Qwen3.8 Flash","deepseek-v4-flash-0731":"DeepSeek V4 Flash","nemotron-3-super-120b-a12b":"Nemotron 120B"}
    b.append(text(10, 14, "Per-call wall p50: compiled judge vs policy-in-context", 10.5, "start", C["ink"], "bold"))
    x0, xw = 96, 200; vmax = 5000; y = 24; rh = 10
    for t in [0,2500,5000]:
        x = x0+xw*t/vmax; b.append(line(x, 22, x, 140)); b.append(text(x, 149, f"{t:,}", 8, "middle", C["mute"]))
    for m in models:
        b.append(text(x0-5, y+rh, short[m], 8.5, "end", C["ink"]))
        for k, (fam, col) in enumerate([("judge", C["ceil"]), ("b", C["ceilB"])]):
            a = L[f"{fam}-{m}"]; yy = y + k*rh; cw = x0 + xw*a["call_wall_p50"]/vmax
            b.append(rect(x0, yy+1.5, cw-x0, rh-3, col)); b.append(text(cw+3, yy+rh-2, f"{a['call_wall_p50']:,} ms · {a['completion_p50']} tok", 7.5, "start", C["mute"]))
        y += rh*2 + 3
    b.append(text(10, H-18, "The judge answers in 5–7 tokens, B in 58–113: the gap is decode volume, not model speed.", 8.5, "start", C["mute"]))
    b.append(text(10, H-6, "Pooled over three passes. At 60 tok/s (a local server) that is ~0.1 s vs ~1–2 s of decode per message.", 8.5, "start", C["mute"]))
    svg(W, H, b, "e-latency.svg")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (f1, f2, f3, f5, f6, f7, f8, e_scatter, e_variance, e_thinkon, e_latency): fn()
    # contact sheet for visual checking
    figs = sorted(f for f in os.listdir(OUT) if f.endswith(".svg"))
    html = "<html><body style='margin:0;background:#fff'>" + "".join(f"<div style='padding:6px;border-bottom:1px solid #ddd'><div style='font:11px monospace;color:#888'>{f}</div><img src='{f}' style='max-width:720px;display:block'></div>" for f in figs) + "</body></html>"
    open(os.path.join(OUT, "_contact.html"), "w").write(html); print("  wrote _contact.html")
