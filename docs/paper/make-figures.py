#!/usr/bin/env python3
"""Every figure in the paper and the one-pager, as SVG, from docs/paper/data/numbers.json.

Pure Python. Colour is assigned by ROLE and the five-hue set was validated with the
dataviz skill's validator (adjacent CVD dE 9.2, normal-vision 27.5, all checks pass):
  aqua    in-browser arms (the shippable 2-4B models)
  orange  model-free floors and oracles (hatched)
  blue    hosted arms, compiled judge, reasoning off
  magenta hosted arms, policy-in-context (Approach B), reasoning off
  violet  reasoning ON
Aqua and magenta sit below 3:1 on the light surface; every mark in them carries a
direct label, which is the relief the validator requires.
"""
import json, os
import statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "data/numbers.json")))
TS = N.get("typesafe")
TSS = N.get("typesafe_scores")
OUT = os.path.join(HERE, "figures")
C = {"local": "#1baf7a", "floor": "#eb6834", "judge": "#2a78d6", "b": "#e87ba4", "think": "#4a3aa7", "ts": "#008300",
     "ink": "#0b0b0b", "mute": "#52514e", "grid": "#e5e7eb", "bg": "#ffffff"}
FONT = "font-family='Helvetica Neue, Helvetica, Arial, sans-serif'"

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;")
def text(x, y, s, size=10, anchor="start", color=C["ink"], weight="normal", rot=None):
    t = f"transform='rotate({rot} {x} {y})'" if rot is not None else ""
    return f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' text-anchor='{anchor}' fill='{color}' font-weight='{weight}' {t}>{esc(s)}</text>"
def rect(x, y, w, h, fill, stroke="none", rx=2, op=1.0):
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w,0):.1f}' height='{max(h,0):.1f}' fill='{fill}' stroke='{stroke}' rx='{rx}' fill-opacity='{op}'/>"
def line(x1, y1, x2, y2, color=C["grid"], w=1, dash=None):
    d = f"stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{w}' {d}/>"
def circ(x, y, r, fill, stroke="white"): return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"
HATCH = ("<defs><pattern id='hatch' patternUnits='userSpaceOnUse' width='6' height='6' patternTransform='rotate(45)'>"
         f"<rect width='6' height='6' fill='#fde8dc'/><line x1='0' y1='0' x2='0' y2='6' stroke='{C['floor']}' stroke-width='1.6'/></pattern></defs>")
def svg(w, h, body, name):
    doc = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img'>" + rect(0, 0, w, h, C["bg"], rx=0)
           + HATCH + "".join(body) + "</svg>")
    open(os.path.join(OUT, name), "w", encoding="utf8").write(doc); print("  wrote", name)
def yaxis(x0, y0, y1, vmax, ticks, label=None, w=None, fmt="{:g}"):
    b = []
    for t in ticks:
        y = y1 - (y1 - y0) * t / vmax
        b.append(line(x0, y, (w or x0 + 6), y)); b.append(text(x0 - 4, y + 3.5, fmt.format(t), 9, "end", C["mute"]))
    if label: b.append(text(x0 - 30, (y0 + y1) / 2, label, 9.5, "middle", C["mute"], rot=-90))
    return b
def bar(b, cx, y1, y, bw, col, hatch=False, label=None, lsize=9.5):
    if hatch: b.append(rect(cx - bw / 2, y, bw, y1 - y, "url(#hatch)", stroke=C["floor"]))
    else: b.append(rect(cx - bw / 2, y, bw, y1 - y, col))
    if label is not None: b.append(text(cx, y - 4, label, lsize, "middle", C["ink"], "bold"))

ML, SW, SL, LAT, PV, FE, CK = (N["message_level"], N["spanwise_predicate"], N["spanlevel_entity"], N["latency"], N["prevention"], N["local_feasibility"], N["cost_per_1k"])
def fe(prefix, model):
    return next(v for k, v in FE.items() if k.startswith(f"{prefix}{model} ["))
def pv_mean(arm):
    v = [x for k, x in PV.items() if k.startswith(arm + " [ceiling-0")]
    return sum(x["leak_prevention"] for x in v) / len(v), sum(x["over_blocking"] for x in v) / len(v)

# ---------------------------------------------------------------- F1 architecture
def f1():
    W, H = 700, 210; b = []
    def box(x, y, w, h, title, sub, fill="#f3f4f6", stroke="#9ca3af"):
        b.append(rect(x, y, w, h, fill, stroke, 4)); b.append(text(x + w / 2, y + 15, title, 9.5, "middle", C["ink"], "bold"))
        for j, s in enumerate(sub): b.append(text(x + w / 2, y + 28 + j * 11, s, 8.3, "middle", C["mute"]))
    def arrow(x1, y1, x2, y2): b.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#6b7280' stroke-width='1.4' marker-end='url(#ah)'/>")
    b.append("<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#6b7280'/></marker></defs>")
    b.append(text(10, 14, "Compile once in the cloud; check every prompt locally", 10.5, "start", C["ink"], "bold"))
    box(10, 26, 118, 60, "Policy document", ["natural language,", "provider clauses", "(p-fin: 5,320 bytes)"])
    box(150, 26, 118, 60, "Compiler", ["frontier model,", "compile-time only —", "touches no user data"], "#fff3ec", C["floor"])
    box(290, 26, 118, 60, "Policy IR", ["8 entity types · 10 rules", "1 semantic predicate (message)", "actions per provider"])
    arrow(128, 56, 150, 56); arrow(268, 56, 290, 56)
    b.append(line(430, 20, 430, 200, "#d1d5db", 1, "4,3")); b.append(text(436, 20, "runtime — in the browser, per prompt", 8.5, "start", C["mute"]))
    box(440, 30, 78, 46, "Tier 0", ["regex + validators", "sync"], "#eef2ff", "#818cf8")
    box(526, 30, 78, 46, "Tier 1", ["span tagger", "ONNX"], "#eef2ff", "#818cf8")
    box(612, 30, 78, 46, "Tier 2", ["LLM judge", "WebLLM / WebGPU"], "#eef2ff", "#818cf8")
    arrow(408, 56, 440, 53); arrow(518, 53, 526, 53); arrow(604, 53, 612, 53)
    box(440, 96, 250, 48, "Cluster-strictest action", ["block > redact > pseudonymize > allow", "pseudonyms rehydrated in the reply"], "#ecfdf5", "#34d399")
    arrow(565, 76, 565, 96)
    b.append(rect(10, 108, 410, 96, "#fafafa", "#d1d5db", 4))
    b.append(text(18, 122, "What this paper varies — the tier-2 judge, two ways to build it, two hardware classes", 9.5, "start", C["ink"], "bold"))
    b.append(text(18, 134, "Compiled judge: one predicate from the IR, 408–447 prompt tokens on the study corpus.", 8, "start", C["mute"]))
    b.append(text(18, 144.5, "Policy-in-context (B): the whole policy in the prompt, ~1,570 tokens.", 8, "start", C["mute"]))
    b.append(rect(18, 148, 8, 8, C["local"])); b.append(text(30, 155, "Q1 · in-browser, shippable: Qwen3.5-2B, Ministral-3-3B, Qwen3-4B, Phi-4-mini", 8))
    b.append(text(30, 165.5, "4-bit weights on WebGPU in real Chrome, 8,192-token context, 5 s per-message budget", 8, "start", C["mute"]))
    b.append(rect(18, 169, 8, 8, C["judge"])); b.append(rect(28, 169, 8, 8, C["b"])); b.append(text(40, 176, "Q2 · hosted, not shippable: five open-weight models (27–120 B where published)", 8))
    b.append(text(40, 186.5, "same judge and same B prompt, providers pinned, three passes at temperature 0", 8, "start", C["mute"]))
    b.append(rect(18, 190, 8, 8, C["think"])); b.append(text(30, 197, "+ GLM-5.3 Flash, mandatory reasoning: thinking ON at 600 and 8,192-token caps, one pass each", 8))
    svg(W, H, b, "f1-architecture.svg")

# ---------------------------------------------------------------- F2 scorecard (paper Figure 3) + one-pager version
def panel_accuracy(b, x0, y0, y1, pw):
    ax = x0 + 36
    b.append(text(x0, y0 - 16, "A  Accuracy — semantic predicate, message-level F1", 10.5, "start", C["ink"], "bold"))
    b += yaxis(ax, y0, y1, 1.0, [0, .25, .5, .75, 1], "F1", w=x0 + pw); b.append(line(ax, y1, x0 + pw, y1, C["mute"]))
    jd = [ML["judge-deepseek-v4-flash-0731"][f"ceiling-0{p}"]["F1"] for p in (1, 2, 3)]
    items = [("in-browser best\n(2–4 B)", N["local_best_message"]["F1"], C["local"], False, None), ("floor: a rule\nwith no model", N["floor_message_level_all"]["F1"], C["floor"], True, None),
             ("hosted best,\nthree passes", sum(jd) / 3, C["judge"], False, jd), ("hosted,\nreasoning on", N["thinkon"]["glm_8192"]["message_attempted"]["F1"], C["think"], False, None)]
    if TS:
        tsp = [TS["predicate_at_half"][p]["F1"] for p in TS["passes"]]
        items.append(("typed judgment\n(TypeSafe)", sum(tsp) / 3, C["ts"], False, tsp))
    step = (pw - 40) / len(items); bw = step * 0.58
    for k, (lab, v, col, hatch, dots) in enumerate(items):
        cx = ax + 4 + step * k + step / 2; y = y1 - (y1 - y0) * v
        ytop = min([y] + [y1 - (y1 - y0) * d for d in (dots or [])])
        bar(b, cx, y1, y, bw, col, hatch, None)
        b.append(text(cx, ytop - (9 if dots else 5), f"{v:.3f}", 9.5, "middle", C["ink"], "bold"))
        if dots:
            for d in dots: b.append(circ(cx, y1 - (y1 - y0) * d, 3.2, "white", col))
        for j, ln in enumerate(lab.split("\n")): b.append(text(cx, y1 + 11 + j * 10, ln, 8, "middle", C["mute"]))

def panel_prevention(b, x0, y0, y1, pw):
    ax = x0 + 36
    b.append(text(x0, y0 - 16, "B  Prevention — leaks fully caught vs clean messages actioned", 10.5, "start", C["ink"], "bold"))
    b += yaxis(ax, y0, y1, 1.0, [0, .25, .5, .75, 1], "", w=x0 + pw, fmt="{:.0%}"); b.append(line(ax, y1, x0 + pw, y1, C["mute"]))
    q = next(v for k, v in PV.items() if k.startswith("tier2-Qwen3.5-2B ["))
    dl, do = pv_mean("b-deepseek-v4-flash-0731"); ql, qo = pv_mean("b-qwen3.8-27b"); g = PV["b-glm-5.3-flash [thinkonglm-01]"]
    groups = [("in-browser\n2 B pipeline", q["leak_prevention"], q["over_blocking"], C["local"]), ("hosted B,\nDeepSeek", dl, do, C["b"]),
              ("hosted B,\nQwen3.8 27B", ql, qo, C["b"]), ("hosted B,\nreasoning on*", g["leak_prevention"], g["over_blocking"], C["think"])]
    if TS:
        tl = st.mean(TS["entity_at_half"][p]["leak_prevention"] for p in TS["passes"])
        to = st.mean(TS["entity_at_half"][p]["over_blocking"] for p in TS["passes"])
        groups.append(("typed judgment\nat 0.5", tl, to, C["ts"]))
    step = (pw - 40) / len(groups); bw = step * 0.27
    for k, (lab, lp, ob, col) in enumerate(groups):
        cx = ax + 4 + step * k + step / 2
        y = y1 - (y1 - y0) * lp; b.append(rect(cx - bw - 1, y, bw, y1 - y, col)); b.append(text(cx - bw / 2 - 1, y - 4, f"{lp:.0%}", 8.5, "middle", C["ink"], "bold"))
        y = y1 - (y1 - y0) * ob; b.append(rect(cx + 1, y, bw, y1 - y, col, op=0.38)); b.append(text(cx + bw / 2 + 1, y - 4, f"{ob:.0%}", 8.5, "middle", C["mute"], "bold"))
        for j, ln in enumerate(lab.split("\n")): b.append(text(cx, y1 + 11 + j * 10, ln, 8, "middle", C["mute"]))
    lx = ax + 8
    b.append(rect(lx, y0 + 2, 8, 8, C["ink"])); b.append(text(lx + 11, y0 + 9, "leaks caught", 7.5, "start", C["mute"]))
    b.append(rect(lx, y0 + 13, 8, 8, C["ink"], op=0.35)); b.append(text(lx + 11, y0 + 20, "clean actioned", 7.5, "start", C["mute"]))

def panel_time(b, x0, y0, pw):
    ax = x0 + 118; bw_ = pw - 118 - 60
    b.append(text(x0, y0 - 16, "C  Time per message, median (ms)", 10.5, "start", C["ink"], "bold"))
    rows = [("in-browser, 2 B", fe("tier2-", "Qwen3.5-2B")["item_wall_p50"], C["local"]), ("in-browser, 4 B", fe("tier2-", "Qwen3-4B")["item_wall_p50"], C["local"]),
            ("in-browser B, 2 B", fe("baselineB-", "Qwen3.5-2B")["item_wall_p50"], C["local"]),
            ("hosted judge, Mistral", LAT["judge-mistral-small-2603"]["item_wall_p50_answered"], C["judge"]), ("hosted judge, DeepSeek", LAT["judge-deepseek-v4-flash-0731"]["item_wall_p50_answered"], C["judge"]),
            ("hosted B, DeepSeek", LAT["b-deepseek-v4-flash-0731"]["item_wall_p50_answered"], C["b"]), ("hosted B, Nemotron", LAT["b-nemotron-3-super-120b-a12b"]["item_wall_p50_answered"], C["b"])]
    if TS: rows.append(("typed judgment", TS["cost_time"]["item_wall_p50"], C["ts"]))
    vmax = 6000; rh = 19; y = y0
    for t in (0, 2500, 5000):
        x = ax + bw_ * t / vmax; b.append(line(x, y - 4, x, y + rh * len(rows) + 2)); b.append(text(x, y + rh * len(rows) + 13, f"{t:,}", 8, "middle", C["mute"]))
    xb = ax + bw_ * 5000 / vmax; b.append(line(xb, y - 6, xb, y + rh * len(rows) + 2, C["floor"], 1.5, "4,3")); b.append(text(xb + 3, y - 1, "5 s budget", 7.5, "start", C["floor"], "bold"))
    for lab, v, col in rows:
        w = bw_ * min(v, vmax) / vmax
        b.append(text(ax - 5, y + 13, lab, 8, "end", C["ink"])); b.append(rect(ax, y + 3, w, rh - 7, col)); b.append(text(ax + w + 3, y + 13, f"{v:,}", 7.5, "start", C["mute"])); y += rh

def panel_cost(b, x0, y0, pw):
    ax = x0 + 92; bw_ = pw - 92 - 46
    b.append(text(x0, y0 - 16, "D  Cost per 1,000 messages (USD)", 10.5, "start", C["ink"], "bold"))
    cm = CK["thinkoff_per_model"]
    short = {"deepseek-v4-flash-0731": "DeepSeek", "mistral-small-2603": "Mistral", "qwen3.8-flash": "Qwen3.8 Flash", "nemotron-3-super-120b-a12b": "Nemotron", "qwen3.8-27b": "Qwen3.8 27B"}
    rows = [("in-browser", 0.0, C["local"])] + [(short[m], v, C["judge"]) for m, v in sorted(cm.items(), key=lambda kv: kv[1])] + [("reasoning on, B", CK["thinkon_glm_b"], C["think"])]
    if TS: rows.insert(2, ("typed judgment", TS["cost_time"]["per_1k_usd"], C["ts"]))
    vmax = 1.5; rh = 19; y = y0
    for t in (0, 0.5, 1.0, 1.5):
        x = ax + bw_ * t / vmax; b.append(line(x, y - 4, x, y + rh * len(rows) + 2)); b.append(text(x, y + rh * len(rows) + 13, f"${t:.2f}", 8, "middle", C["mute"]))
    for lab, v, col in rows:
        w = bw_ * v / vmax
        b.append(text(ax - 5, y + 13, lab, 8, "end", C["ink"]))
        if v > 0: b.append(rect(ax, y + 3, w, rh - 7, col)); b.append(text(ax + w + 3, y + 13, f"${v:.2f}", 7.5, "start", C["mute"]))
        else: b.append(text(ax + 2, y + 13, "$0 — the user's own GPU", 7.5, "start", C["local"], "bold"))
        y += rh
    b.append(text(ax, y + 27, f"judge family ${CK['thinkoff_judge']:.2f} · policy-in-context ${CK['thinkoff_b']:.2f} per 1,000 (thinking off)", 7.5, "start", C["mute"]))

def f2():
    W, H = 720, 476; b = []
    panel_accuracy(b, 10, 30, 190, 340); panel_prevention(b, 370, 30, 190, 340)
    panel_time(b, 10, 262, 340); panel_cost(b, 370, 262, 340)
    b.append(text(10, H - 6, "Same corpus, policy and gold throughout. Hatched = a rule with no model. B = policy-in-context. Typed judgment = TypeSafe at its default 0.5. *Reasoning-on counts 59 unanswered rows as misses.", 8, "start", C["mute"]))
    svg(W, H, b, "f2-scorecard.svg")

def e_scorecard():
    W, H = 720, 212; b = []
    panel_accuracy(b, 10, 26, 158, 340); panel_prevention(b, 370, 26, 158, 340)
    b.append(text(10, H - 6, "Same corpus, policy and gold. Hatched = a rule with no model. B = policy-in-context. Typed judgment = TypeSafe at 0.5. *Reasoning-on counts its 59 unanswered rows as misses.", 8, "start", C["mute"]))
    svg(W, H, b, "e-scorecard.svg")

# ---------------------------------------------------------------- F3 in-browser feasibility (paper Figure 2)
def f3():
    W, H = 700, 232; b = []
    b.append(text(10, 16, "In the browser: per-message wall time (bar = median, tick = p95) against the policy's 5-second budget", 10.5, "start", C["ink"], "bold"))
    rows = [("compiled · Qwen3.5-2B", fe("tier2-", "Qwen3.5-2B")), ("compiled · Ministral-3-3B", fe("tier2-", "Ministral-3-3B")), ("compiled · Phi-4-mini", fe("tier2-", "Phi-4-mini-instruct")),
            ("compiled · Qwen3-4B", fe("tier2-", "Qwen3-4B")), ("B · Qwen3.5-2B", fe("baselineB-", "Qwen3.5-2B")), ("B · Ministral-3-3B", fe("baselineB-", "Ministral-3-3B")),
            ("B · Phi-4-mini", fe("baselineB-", "Phi-4-mini-instruct")), ("B · Qwen3-4B", fe("baselineB-", "Qwen3-4B"))]
    ax, pw = 150, 330; vmax = 9000; rh = 20; y = 34
    for t in (0, 2500, 5000, 7500):
        x = ax + pw * t / vmax; b.append(line(x, 28, x, y + rh * len(rows) + 2)); b.append(text(x, y + rh * len(rows) + 13, f"{t:,} ms", 8, "middle", C["mute"]))
    xb = ax + pw * 5000 / vmax; b.append(line(xb, 24, xb, y + rh * len(rows) + 2, C["floor"], 1.6, "4,3")); b.append(text(xb + 3, 24, "5,000 ms budget", 8, "start", C["floor"], "bold"))
    ax2, pw2 = 540, 120
    b.append(text(ax2, 28, "answered inside budget", 8, "start", C["mute"]))
    for lab, v in rows:
        p50, p95 = v["item_wall_p50"], v["item_wall_p95"]
        b.append(text(ax - 4, y + 13, lab, 8.5, "end", C["ink"]))
        b.append(rect(ax, y + 3, pw * min(p50, vmax) / vmax, rh - 7, C["local"]))
        xp = ax + pw * min(p95, vmax) / vmax; b.append(line(xp, y + 1, xp, y + rh - 3, C["ink"], 1.8)); b.append(text(min(xp, ax + pw) + 4, y + 13, f"{p50:,} / {p95:,}", 7.5, "start", C["mute"]))
        share = min(v["answered"], v["items"]) / v["items"]
        b.append(rect(ax2, y + 4, pw2, rh - 9, "#eef2f7", rx=1)); b.append(rect(ax2, y + 4, pw2 * share, rh - 9, C["local"] if share == 1 else "#7fd3b3", rx=1))
        b.append(text(ax2 + pw2 + 4, y + 13, f"{min(v['answered'], v['items'])}/{v['items']}", 7.5, "start", C["ink"], "bold" if share == 1 else "normal")); y += rh
    b.append(text(10, H - 6, "Wall time is tier-2 time per message in real Chrome on WebGPU. A message the judge could not answer inside the budget spends the whole budget and is scored as a miss.", 8, "start", C["mute"]))
    svg(W, H, b, "f3-feasibility.svg")

# ---------------------------------------------------------------- F4 variance (paper Figure 7, wide)
def f4():
    W, H = 700, 300; b = []
    sp = N["variance"]["per_arm"]; arms = sorted(sp, key=lambda a: -sum(sp[a]["passes"]) / 3)
    x0, y0, y1 = 52, 30, 240; pw = 470
    b.append(text(10, 16, "A  Message-level predicate F1, every hosted arm, three passes at temperature 0", 10.5, "start", C["ink"], "bold"))
    b += yaxis(x0, y0, y1, 1.0, [0, .25, .5, .75, 1], "F1", w=x0 + pw)
    fl = N["floor_message_level_all"]["F1"]; yf = y1 - (y1 - y0) * fl
    b.append(line(x0, yf, x0 + pw, yf, C["floor"], 1.5, "4,3")); b.append(text(x0 + pw - 2, yf + 11, f"floor {fl:.3f}", 9, "end", C["floor"], "bold"))
    step = pw / len(arms)
    for k, a in enumerate(arms):
        cx = x0 + step * k + step / 2; v = sp[a]["passes"]; col = C["judge"] if a.startswith("judge") else C["b"]
        b.append(line(cx, y1 - (y1 - y0) * min(v), cx, y1 - (y1 - y0) * max(v), col, 2))
        for d in v: b.append(circ(cx, y1 - (y1 - y0) * d, 3.2, col))
        if sp[a]["spread"] > 0.05: b.append(text(cx, y1 - (y1 - y0) * max(v) - 7, f"{sp[a]['spread']:.2f}", 8, "middle", C["mute"]))
        lab = a.replace("-v4-flash-0731", "").replace("-3-super-120b-a12b", "").replace("-small-2603", "").replace("qwen3.8-", "qwen-").replace("judge-", "J·").replace("b-", "B·")
        b.append(text(cx + 3, y1 + 9, lab, 8.5, "end", C["mute"], rot=-40))
    bx = 560; b.append(text(bx - 8, 16, "B  Pass-to-pass spread", 10.5, "start", C["ink"], "bold"))
    b += yaxis(bx + 20, y0, y1, 0.4, [0, .1, .2, .3, .4], "F1 spread, mean (dot = max)", w=bx + 130)
    for k, (lab, m, mx, col) in enumerate([("entity\nspans", N["variance"]["entity_mean_spread"], N["variance"]["entity_max_spread"], C["b"]), ("predicate,\nmessage level", N["variance"]["mean_spread"], N["variance"]["max_spread"], C["judge"])]):
        cx = bx + 45 + k * 55; y = y1 - (y1 - y0) * m / 0.4
        b.append(rect(cx - 16, y, 32, y1 - y, col)); b.append(circ(cx, y1 - (y1 - y0) * mx / 0.4, 3.5, col))
        b.append(text(cx, y - 5, f"{m:.3f}", 9.5, "middle", C["ink"], "bold"))
        for j, ln in enumerate(lab.split("\n")): b.append(text(cx, y1 + 11 + j * 10, ln, 8.5, "middle", C["mute"]))
    b.append(text(10, H - 6, "Providers pinned, temperature 0 — what moves between passes is routing, not the sampler. Number above a stem = spread > 0.05. J = judge, B = policy-in-context.", 8, "start", C["mute"]))
    svg(W, H, b, "f4-variance.svg")

# ---------------------------------------------------------------- F5 thinking-ON (paper Figure 4)
def f5():
    W, H = 720, 256; b = []
    t6, t8 = N["thinkon"]["glm_600"], N["thinkon"]["glm_8192"]
    b.append(text(10, 16, "A  GLM judge, reasoning on: 600 vs 8,192-token cap (message level, answered rows)", 10.5, "start", C["ink"], "bold"))
    x0, y0, y1 = 48, 32, 190; pw = 330
    b += yaxis(x0, y0, y1, 1.0, [0, .25, .5, .75, 1], "", w=x0 + pw); b.append(line(x0, y1, x0 + pw, y1, C["mute"]))
    gs = pw / 3; light = "#b7aee0"
    for g, (lab, key) in enumerate([("precision", "P"), ("recall", "R"), ("F1", "F1")]):
        for k, (blk, col, name) in enumerate([(t6, light, "600"), (t8, C["think"], "8,192")]):
            v = blk["message_attempted"][key]; cx = x0 + gs * g + gs * (0.3 + 0.4 * k); bw = gs * 0.3
            y = y1 - (y1 - y0) * v; b.append(rect(cx - bw / 2, y, bw, y1 - y, col)); b.append(text(cx, y + 11, f"{v:.3f}", 8.5, "middle", "white", "bold"))
            if g == 0: b.append(text(cx, y1 + 22, name, 8.5, "middle", C["mute"]))
        b.append(text(x0 + gs * g + gs / 2, y1 + 11, lab, 9.5, "middle", C["ink"]))
        if key == "F1":
            for blk in (t6, t8):
                yf = y1 - (y1 - y0) * blk["floor_attempted"]["F1"]; b.append(line(x0 + gs * g + 4, yf, x0 + gs * g + gs - 4, yf, C["floor"], 1.5, "4,3"))
    bx = 440; b.append(text(bx, 16, "B  The gold positives, by call outcome", 10.5, "start", C["ink"], "bold"))
    cats = [("detected_clean", "detected, call finished", C["think"]), ("detected_truncated", "detected, call truncated", light), ("missed_truncated", "MISSED, call truncated", C["floor"]), ("missed_clean", "missed, call finished", "#c9c9c9")]
    for k, (blk, name) in enumerate([(t6, "600 cap"), (t8, "8,192 cap")]):
        cx = bx + 40 + k * 88; y = y1; bw = 50
        for key, lab, col in cats:
            n = blk["split"][key]; h = (y1 - y0) * n / 19
            if n: b.append(rect(cx - bw / 2, y - h, bw, h - 1.5, col)); b.append(text(cx, y - h / 2 + 3.5, str(n), 9.5, "middle", "white" if col != "#c9c9c9" else C["ink"], "bold")); y -= h
        b.append(text(cx, y1 + 11, name, 9.5, "middle", C["ink"])); b.append(text(cx, y1 + 22, f"{blk['positives_answered']} of 19 answered", 8.5, "middle", C["mute"]))
    for j, (key, lab, col) in enumerate(cats):
        b.append(rect(bx + 172, 40 + j * 14, 9, 9, col)); b.append(text(bx + 185, 48 + j * 14, lab, 8, "start", C["mute"]))
    b.append(text(bx, 228, f"truncated rows {t6['truncated_rows']} → {t8['truncated_rows']} of 189;  unanswered (429s) {189 - t6['answered']} → {189 - t8['answered']}", 8.5, "start", C["mute"]))
    b.append(text(10, H - 6, f"Dashed: floors recomputed over the same answered rows ({t6['floor_attempted']['F1']:.3f} at 600, {t8['floor_attempted']['F1']:.3f} at 8,192). Every miss at 600 was a truncated call, with no clean misses; at 8,192 the missed list is empty.", 8.5, "start", C["mute"]))
    svg(W, H, b, "f5-thinking-on.svg")

# ---------------------------------------------------------------- F6 hosted latency (paper Figure 5)
def f6():
    W, H = 700, 250; b = []
    models = ["mistral-small-2603", "qwen3.8-27b", "qwen3.8-flash", "deepseek-v4-flash-0731", "nemotron-3-super-120b-a12b"]
    short = {"mistral-small-2603": "Mistral Small", "qwen3.8-27b": "Qwen3.8 27B", "qwen3.8-flash": "Qwen3.8 Flash", "deepseek-v4-flash-0731": "DeepSeek V4 Flash", "nemotron-3-super-120b-a12b": "Nemotron 3 Super 120B"}
    b.append(text(10, 16, "Hosted, per-call wall time p50 (ms), pooled over three passes — compiled judge vs policy-in-context on the same model", 10.5, "start", C["ink"], "bold"))
    x0, xw = 150, 440; vmax = 5000; y = 34; rh = 17
    for t in [0, 1000, 2000, 3000, 4000, 5000]:
        x = x0 + xw * t / vmax; b.append(line(x, 32, x, 226)); b.append(text(x, 29, f"{t:,} ms", 8, "middle", C["mute"]))
    for m in models:
        b.append(text(x0 - 6, y + rh - 2, short[m], 9.5, "end", C["ink"]))
        for k, (fam, col, dark) in enumerate([("judge", C["judge"], "#1c5cab"), ("b", C["b"], "#c2537f")]):
            a = LAT[f"{fam}-{m}"]; yy = y + k * rh
            tt = x0 + xw * a["ttft_p50"] / vmax; cw = x0 + xw * a["call_wall_p50"] / vmax
            b.append(rect(x0, yy + 2, cw - x0, rh - 5, col)); b.append(rect(x0, yy + 2, tt - x0, rh - 5, dark))
            b.append(text(cw + 4, yy + rh - 3, f"{a['call_wall_p50']:,} ms · {a['completion_p50']} tok" + (f" · {a['n429']} ×429" if a["n429"] else ""), 8.5, "start", C["mute"]))
        y += rh * 2 + 8
    b.append(rect(x0, 240, 9, 9, "#1c5cab")); b.append(text(x0 + 13, 248, "judge: time to first token", 8.5)); b.append(rect(x0 + 140, 240, 9, 9, C["judge"])); b.append(text(x0 + 153, 248, "judge: decode", 8.5))
    b.append(rect(x0 + 240, 240, 9, 9, "#c2537f")); b.append(text(x0 + 253, 248, "B: first token", 8.5)); b.append(rect(x0 + 330, 240, 9, 9, C["b"])); b.append(text(x0 + 343, 248, "B: decode", 8.5))
    b.append(text(x0 + 420, 248, "tok = completion tokens p50", 8.5, "start", C["mute"]))
    svg(W, H, b, "f6-latency.svg")

# ---------------------------------------------------------------- F8 precision/recall scatter (paper Figure 6)
def f9():
    """The judgment arm's two distinctive pictures: a separating score, and a knob."""
    W, H = 720, 310; b = []
    x0, x1, ax = 20, 400, 62
    b.append(text(x0, 20, "A  Every message's probability, by what the annotators said", 10.5, "start", C["ink"], "bold"))
    def X(p): return ax + (x1 - ax) * p
    for t in (0, .25, .5, .75, 1):
        b.append(line(X(t), 34, X(t), 212)); b.append(text(X(t), 226, f"{t:g}", 8.5, "middle", C["mute"]))
    b.append(text((ax + x1) / 2, 243, "P(the message discloses a client relationship)", 9, "middle", C["mute"]))
    hi, lo = TSS["highest_negative"], TSS["lowest_positive"]
    b.append(rect(X(hi), 34, X(lo) - X(hi), 178, "#f1f5f9"))
    b.append(line(X(hi), 34, X(hi), 212, C["mute"], 1, "3,3")); b.append(line(X(lo), 34, X(lo), 212, C["mute"], 1, "3,3"))
    b.append(text((X(hi) + X(lo)) / 2, 30, "empty band", 8, "middle", C["ink"], "bold"))
    b.append(text((X(hi) + X(lo)) / 2, 256, f"no message scores between {hi:g} and {lo:g}", 8, "middle", C["ink"]))
    lanes = ((f"{len(TSS['negatives'])} clean", TSS["negatives"], C["mute"]), (f"{len(TSS['positives'])} leak", TSS["positives"], C["ts"]))
    for lane, (label, vals, col) in enumerate(lanes):
        cy = 78 + lane * 86
        b.append(text(ax - 8, cy + 3, label, 8.5, "end", C["ink"]))
        for i, v in enumerate(vals):
            b.append(f"<circle cx='{X(v):.1f}' cy='{cy + (i % 7) * 4 - 12:.1f}' r='2.6' fill='{col}' fill-opacity='0.72'/>")
    b.append(line(X(0.5), 34, X(0.5), 212, C["floor"], 1.5, "4,3")); b.append(text(X(0.5) + 4, 46, "default 0.5", 8, "start", C["floor"], "bold"))
    px0, py0, px1, py1 = 470, 40, 700, 212
    b.append(text(440, 20, "B  Leaks stopped against clean prompts flagged", 10.5, "start", C["ink"], "bold"))
    def PX(v): return px0 + (px1 - px0) * min(v, 0.6) / 0.6
    def PY(v): return py1 - (py1 - py0) * v
    for t in (0, .2, .4, .6):
        b.append(line(PX(t), py0, PX(t), py1)); b.append(text(PX(t), py1 + 12, f"{t:.0%}", 8.5, "middle", C["mute"]))
    for t in (0, .25, .5, .75, 1):
        b.append(line(px0, PY(t), px1, PY(t))); b.append(text(px0 - 5, PY(t) + 3, f"{t:.0%}", 8.5, "end", C["mute"]))
    b.append(text((px0 + px1) / 2, py1 + 25, "clean messages flagged", 9, "middle", C["mute"]))
    b.append(text(px0 - 36, (py0 + py1) / 2, "leaks stopped", 9, "middle", C["mute"], rot=-90))
    pts = [f"{PX(x['over_blocking']):.1f},{PY(x['leak_prevention']):.1f}" for x in sorted(TS["entity_sweep"], key=lambda z: z["over_blocking"])]
    b.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='{C['ts']}' stroke-width='2'/>")
    for t, lab in ((0.5, "0.5"), (0.85, "0.85"), (0.95, "0.95")):
        pt = next(x for x in TS["entity_sweep"] if abs(x["threshold"] - t) < 1e-9)
        b.append(circ(PX(pt["over_blocking"]), PY(pt["leak_prevention"]), 4, C["ts"]))
        b.append(text(PX(pt["over_blocking"]) + 7, PY(pt["leak_prevention"]) + 3, lab, 8, "start", C["ts"], "bold"))
    t0 = TS["entity_tier0"]; q = next(v for k, v in PV.items() if k.startswith("tier2-Qwen3.5-2B ["))
    dl, do = pv_mean("b-deepseek-v4-flash-0731"); ql, qo = pv_mean("b-qwen3.8-27b")
    for x, y, col, lab, dy in ((t0["over_blocking"], t0["leak_prevention"], C["floor"], "regex tier alone", -9),
                               (q["over_blocking"], q["leak_prevention"], C["local"], "in-browser", 16),
                               (do, dl, C["b"], "hosted B, DeepSeek", -9), (qo, ql, C["b"], "hosted B, Qwen 27B", -9)):
        b.append(f"<rect x='{PX(x) - 3.5:.1f}' y='{PY(y) - 3.5:.1f}' width='7' height='7' fill='{col}' stroke='white'/>")
        b.append(text(PX(x) - 6, PY(y) + dy, lab, 7.5, "end", col, "bold"))
    b.append(text(px1, py0 - 6, "better: up and left", 7.5, "end", C["mute"]))
    b.append(text(20, H - 8, "Left: pass 1, all 179 scored messages, dots stacked where they collide. Right: the judgment threshold swept 0 to 1 (line) against every other arm's single operating point (squares).", 8, "start", C["mute"]))
    svg(W, H, b, "f9-typesafe.svg")

def f8():
    W, H = 340, 316; b = []
    x0, y0, x1, y1 = 40, 40, 325, 272
    def X(p): return x0 + (x1 - x0) * p
    def Y(r): return y1 - (y1 - y0) * r
    b.append(text(10, 14, "Message-level predicate: precision vs recall, every arm", 10.5, "start", C["ink"], "bold"))
    for t in [0, .25, .5, .75, 1.0]:
        b.append(line(X(t), y0, X(t), y1)); b.append(text(X(t), y1 + 11, f"{t:g}", 8.5, "middle", C["mute"]))
        b.append(line(x0, Y(t), x1, Y(t))); b.append(text(x0 - 4, Y(t) + 3, f"{t:g}", 8.5, "end", C["mute"]))
    b.append(text((x0 + x1) / 2, y1 + 23, "precision", 9.5, "middle", C["mute"])); b.append(text(x0 - 28, (y0 + y1) / 2, "recall", 9.5, "middle", C["mute"], rot=-90))
    for f in [0.3, 0.5, 0.7, 0.9]:
        pts = []
        for i in range(1, 100):
            p = i / 100; r = f * p / (2 * p - f) if 2 * p - f > 0 else None
            if r and 0 < r <= 1: pts.append(f"{X(p):.1f},{Y(r):.1f}")
        if pts: b.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='#d1d5db' stroke-width='0.8'/>")
        b.append(text(x1 - 2 if f >= 0.9 else X(min(1, f / (2 - f)) + 0.02), Y(0.9) + 10 if f >= 0.9 else Y(f) - 2, f"F1={f}", 7.5, "end", "#9ca3af"))
    for k, v in N["local_message_level"].items(): b.append(circ(X(v["P"]), Y(v["R"]), 3, C["local"], "white"))
    for arm, byrun in ML.items():
        for run, v in byrun.items():
            if run.startswith("ceiling-0"): b.append(circ(X(v["P"]), Y(v["R"]), 3.4, C["judge"] if arm.startswith("judge") else C["b"]))
    fl = N["floor_message_level_all"]; b.append(f"<rect x='{X(fl['P']) - 4.5:.1f}' y='{Y(fl['R']) - 4.5:.1f}' width='9' height='9' fill='{C['floor']}' stroke='white'/>")
    b.append(text(X(fl["P"]) - 7, Y(fl["R"]) + 4, "floor", 8.5, "end", C["floor"], "bold"))
    t8 = N["thinkon"]["glm_8192"]["message_attempted"]; b.append(f"<polygon points='{X(t8['P']):.1f},{Y(t8['R']) - 6:.1f} {X(t8['P']) + 5.5:.1f},{Y(t8['R']) + 4:.1f} {X(t8['P']) - 5.5:.1f},{Y(t8['R']) + 4:.1f}' fill='{C['think']}' stroke='white'/>")
    b.append(text(X(t8["P"]) - 8, Y(t8["R"]) + 34, "reasoning on (8,192)", 8, "end", C["think"], "bold"))
    jd = ML["judge-deepseek-v4-flash-0731"]["ceiling-03"]; b.append(text(X(jd["P"]) - 8, Y(jd["R"]) + 22, "hosted judge, DeepSeek ×3", 8, "end", C["judge"], "bold"))
    if TS:
        for p in TS["passes"]:
            v = TS["predicate_at_half"][p]; b.append(circ(X(v["P"]), Y(v["R"]), 3.6, C["ts"]))
        v = TS["predicate_at_half"][TS["passes"][0]]
        b.append(text(X(v["P"]) - 8, Y(v["R"]) - 8, "typed judgment ×3", 8, "end", C["ts"], "bold"))
    b.append(text(X(0.05), Y(0.62), "in-browser 2–4 B: low precision at any recall", 8, "start", C["local"]))
    b.append(text(10, H - 6, f"Aqua: {len(N['local_message_level'])} in-browser arms · blue: hosted judge · pink: hosted B (3 passes) · grey: iso-F1", 8.5, "start", C["mute"]))
    svg(W, H, b, "f8-pr-scatter.svg")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (f1, f2, e_scorecard, f3, f4, f5, f6, f8, f9): fn()
    figs = sorted(f for f in os.listdir(OUT) if f.endswith(".svg") and not f.startswith(("e-", "f7-")))
    html = "<html><body style='margin:0;background:#fff'>" + "".join(f"<div style='padding:6px;border-bottom:1px solid #ddd'><div style='font:11px monospace;color:#888'>{f}</div><img src='{f}' style='max-width:720px;display:block'></div>" for f in figs) + "</body></html>"
    open(os.path.join(OUT, "_contact.html"), "w").write(html); print("  wrote _contact.html")
