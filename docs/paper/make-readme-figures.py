#!/usr/bin/env python3
"""The two README charts, in light and dark variants, from data/numbers.json.

Palette: the dataviz reference instance. Aqua = small model in the browser, blue =
bigger hosted model; the pair validates in both modes (validate_palette.js). The
no-AI rule is a reference line, not a series, so it wears secondary ink. Every
value is also printed as a direct label and repeated in the README table.
"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "data/numbers.json")))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets"))
THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7", local="#1baf7a", hosted="#2a78d6"),
    "dark":  dict(surface="#1a1a19", border="#383835", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835", local="#199e70", hosted="#3987e5"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;")
def text(x, y, s, size, fill, anchor="start", weight=400):
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{esc(s)}</text>"
def line(x1, y1, x2, y2, stroke, w=1, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' stroke-width='{w}'{d}/>"
def hbar(x, y, w, h, fill, r=4):   # square at the baseline, 4px rounded data-end
    r = min(r, w / 2, h / 2)
    return (f"<path d='M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} "
            f"A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z' fill='{fill}'/>")
def swatch(x, y, fill): return f"<rect x='{x}' y='{y}' width='12' height='12' rx='3' fill='{fill}'/>"
def svg(W, H, t, body, label):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' role='img' aria-label='{esc(label)}' font-family=\"{FONT}\">"
            f"<title>{esc(label)}</title><rect x='0.5' y='0.5' width='{W - 1}' height='{H - 1}' rx='12' fill='{t['surface']}' stroke='{t['border']}'/>"
            + "".join(body) + "</svg>\n")

# ---- chart 1: does the model understand the policy? ---------------------------
ml = N["message_level"]
NAMES = {"judge-deepseek-v4-flash-0731": "DeepSeek V4 Flash", "judge-qwen3.8-27b": "Qwen3.8 27B", "judge-qwen3.8-flash": "Qwen3.8 Flash",
         "judge-nemotron-3-super-120b-a12b": "Nemotron 3 Super 120B", "judge-mistral-small-2603": "Mistral Small"}
ROWS = [(NAMES[a], st.mean(ml[a][f"ceiling-0{p}"]["F1"] for p in (1, 2, 3)), "hosted") for a in NAMES]
ROWS.append(("Best small model, in browser", N["local_best_message"]["F1"], "local"))
ROWS.sort(key=lambda r: -r[1])
FLOOR = N["floor_message_level_all"]["F1"]

def understanding(t):
    W, pad = 760, 28
    b = [text(pad, 42, "Can the model tell a real client leak from a harmless mention?", 17, t["ink"], weight=600),
         text(pad, 64, "Score from 0 to 1 (F1, higher is better) on 189 test messages. Hosted models: average of 3 runs.", 13, t["ink2"]),
         swatch(pad, 82, t["local"]), text(pad + 18, 93, "Small model, in the browser", 12.5, t["ink2"]),
         swatch(pad + 220, 82, t["hosted"]), text(pad + 238, 93, "Bigger model, hosted", 12.5, t["ink2"])]
    top, pitch, bh = 146, 36, 20
    x0, x1 = 236, W - 64; pw = x1 - x0
    bottom = top + pitch * len(ROWS)
    for v in (0, .25, .5, .75, 1):
        b.append(line(x0 + v * pw, top - 4, x0 + v * pw, bottom, t["base"] if v == 0 else t["grid"]))
        b.append(text(x0 + v * pw, bottom + 18, f"{v:g}", 11.5, t["muted"], "middle"))
    for i, (name, v, kind) in enumerate(ROWS):
        yc = top + pitch * i + pitch / 2
        b.append(text(pad, yc + 4.5, name, 13, t["ink"]))
        b.append(hbar(x0, yc - bh / 2, v * pw, bh, t[kind]))
        b.append(text(x0 + v * pw + 10, yc + 4.5, f"{v:.2f}", 13, t["ink"], weight=600))
    xr = x0 + FLOOR * pw
    b.append(line(xr, top - 14, xr, bottom, t["ink2"], 1.5, "5,4"))
    b.append(text(xr, top - 20, f"No-AI rule scores {FLOOR:.2f}: the line to beat", 12.5, t["ink"], "middle", 600))
    y = bottom + 50
    b.append(text(pad, y, "No-AI rule = flag every message that contains a capitalised company name.", 11.5, t["muted"]))
    b.append(text(pad, y + 17, "It scores well because every leak in this test set names a company. A model must beat it to be worth running.", 11.5, t["muted"]))
    label = ("Bar chart of policy-understanding score (F1). " + "; ".join(f"{n} {v:.2f}" for n, v, _ in ROWS) + f". No-AI rule {FLOOR:.2f}.")
    return svg(W, y + 36, t, b, label)

# ---- chart 2: leaks stopped vs safe prompts flagged -----------------------------
PV = N["prevention"]
LOC = next(v for k, v in PV.items() if k.startswith("tier2-Qwen3.5-2B ["))
DS = [v for k, v in PV.items() if k.startswith("b-deepseek-v4-flash-0731 [ceiling-0")]
assert len(DS) == 3
GROUPS = [("Leaks stopped", "higher is better", 100 * LOC["leak_prevention"], 100 * st.mean(d["leak_prevention"] for d in DS)),
          ("Safe prompts flagged by mistake", "lower is better", 100 * LOC["over_blocking"], 100 * st.mean(d["over_blocking"] for d in DS))]

def prevention(t):
    W, pad = 760, 28
    b = [text(pad, 42, "How many leaks get stopped, and how many safe prompts get flagged?", 17, t["ink"], weight=600),
         text(pad, 64, "Out of every 100 messages of each kind, on 189 test messages. Hosted model: average of 3 runs.", 13, t["ink2"]),
         swatch(pad, 82, t["local"]), text(pad + 18, 93, "Small model, in the browser (Qwen3.5 2B)", 12.5, t["ink2"]),
         swatch(pad + 300, 82, t["hosted"]), text(pad + 318, 93, "Bigger model, hosted (DeepSeek V4 Flash)", 12.5, t["ink2"])]
    x0, x1 = 150, W - 100; pw = x1 - x0
    y = 134
    for gi, (head, hint, lv, hv) in enumerate(GROUPS):
        b.append(text(pad, y, head, 14, t["ink"], weight=600))
        b.append(text(pad + (len(head) * 7.4) + 12, y, hint, 12, t["muted"]))
        g0, g1 = y + 12, y + 72
        for v in (0, 25, 50, 75, 100):
            b.append(line(x0 + v / 100 * pw, g0, x0 + v / 100 * pw, g1, t["base"] if v == 0 else t["grid"]))
        for ri, (who, v, kind) in enumerate((("In browser", lv, "local"), ("Hosted", hv, "hosted"))):
            yc = g0 + 16 + ri * 28
            b.append(text(pad, yc + 4.5, who, 13, t["ink2"]))
            b.append(hbar(x0, yc - 10, v / 100 * pw, 20, t[kind]))
            b.append(text(x0 + v / 100 * pw + 10, yc + 4.5, f"{round(v)} of 100", 13, t["ink"], weight=600))
        y = g1 + 34
    for v in (0, 25, 50, 75, 100):
        b.append(text(x0 + v / 100 * pw, y - 16, f"{v}", 11.5, t["muted"], "middle"))
    b.append(text(pad, y + 14, "Stopped = every confidential item in the message was caught. In the browser, the pattern rules do most of that work.", 11.5, t["muted"]))
    label = ("Bar chart. " + "; ".join(f"{h}: in browser {round(l)} of 100, hosted {round(hv)} of 100" for h, _, l, hv in GROUPS) + ".")
    return svg(W, y + 34, t, b, label)

for mode, t in THEMES.items():
    for name, fn in (("understanding", understanding), ("prevention", prevention)):
        p = os.path.join(OUT, f"readme-{name}-{mode}.svg"); open(p, "w").write(fn(t)); print("wrote", os.path.relpath(p))
print("rows:", [(n, round(v, 3)) for n, v, _ in ROWS], "floor", FLOOR)
print("prevention:", [(h, round(l, 1), round(hv, 1)) for h, _, l, hv in GROUPS])
