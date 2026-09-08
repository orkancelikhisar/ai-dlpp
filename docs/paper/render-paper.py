#!/usr/bin/env python3
"""Renders docs/paper/paper.html from paper.template.html + data/numbers.json + figures/*.svg.

Every table is generated from numbers.json so it cannot be mistranscribed; the
figures are inlined so the PDF is self-contained. Prose numbers live in the
template and are checked against numbers.json by the reviewer.
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "data/numbers.json")))
T = open(os.path.join(HERE, "paper.template.html"), encoding="utf8").read()

def fig(name):
    s = open(os.path.join(HERE, "figures", name), encoding="utf8").read()
    return re.sub(r"<svg ", "<svg style='width:100%;height:auto' ", s, count=1)

def short(arm):
    return (arm.replace("-v4-flash-0731", "").replace("-3-super-120b-a12b", "").replace("-small-2603", "")
               .replace("qwen3.8-", "qwen-").replace("judge-", "J·").replace("b-", "B·"))

# ---- Table: every hosted arm, three passes, three metrics --------------------
sw, ml, sl = N["spanwise_predicate"], N["message_level"], N["spanlevel_entity"]
arms = ["judge-deepseek-v4-flash-0731", "judge-qwen3.8-27b", "judge-qwen3.8-flash", "judge-nemotron-3-super-120b-a12b", "judge-mistral-small-2603",
        "b-nemotron-3-super-120b-a12b", "b-deepseek-v4-flash-0731", "b-qwen3.8-27b", "b-mistral-small-2603", "b-qwen3.8-flash"]
rows = []
for a in arms:
    swv = [sw.get(f"ceiling-{a} [ceiling-0{p}]", {}).get("F1", "—") for p in (1, 2, 3)]
    mlv = [f"{ml[a][f'ceiling-0{p}']['F1']:.3f}" for p in (1, 2, 3)]
    slv = [sl.get(f"ceiling-{a} [ceiling-0{p}]", {}).get("F1", "—") for p in (1, 2, 3)] if a.startswith("b-") else ["—"] * 3
    un = [N["unanswered"].get(f"{a} [ceiling-0{p}]", {}).get("unanswered", 0) for p in (1, 2, 3)]
    def cell(v, floor):
        try: x = float(v)
        except: return f"<td>{v}</td>"
        return f"<td class='{'up' if x > floor else ''}'>{x:.3f}</td>"
    rows.append("<tr><td class='l'>" + short(a) + "</td>" + "".join(cell(v, 0.571) for v in swv) + "".join(cell(v, 0.776) for v in mlv)
                + "".join(cell(v, 0.695) for v in slv) + f"<td class='mute'>{'/'.join(str(u) for u in un)}</td></tr>")
TABLE_CEILING = ("<table class='data'>{CAP}<thead><tr><th class='l'>arm</th><th colspan='3'>predicate, span-wise (floor 0.571)</th>"
                 "<th colspan='3'>predicate, message level (floor 0.776)</th><th colspan='3'>entity spans (oracle 0.695 / 0.454)</th><th>unanswered</th></tr>"
                 "<tr><th></th>" + "<th>p1</th><th>p2</th><th>p3</th>" * 3 + "<th>p1/p2/p3</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

# ---- Table: latency, pooled over three passes ---------------------------------
L = N["latency"]
lrows = []
for a in sorted(L, key=lambda k: L[k]["call_wall_p50"]):
    x = L[a]
    lrows.append(f"<tr><td class='l'>{short(a)}</td><td>{x['calls']}</td><td>{x['ttft_p50']:,}</td><td>{x['ttft_p95']:,}</td><td><b>{x['call_wall_p50']:,}</b></td>"
                 f"<td>{x['item_wall_p50_answered']:,}</td><td>{x['completion_p50']}</td><td>{x['completion_p50']/60:.2f}</td><td>{x['n429']}</td></tr>")
TABLE_LATENCY = ("<table class='data'>{CAP}<thead><tr><th class='l'>arm</th><th>calls</th><th>TTFT p50</th><th>TTFT p95</th><th>call wall p50</th><th>item wall p50*</th>"
                 "<th>compl. tok p50</th><th>s @60 tok/s</th><th>429s</th></tr></thead><tbody>" + "".join(lrows) + "</tbody></table>")

# ---- Table: spend ------------------------------------------------------------
S = N["spend"]
srows = "".join(f"<tr><td class='l'>{s['segment']}</td><td>{s['calls']:,}</td><td>${s['costUsd']:.5f}</td></tr>" for s in S["segments"])
TABLE_SPEND = ("<table class='data'>{CAP}<thead><tr><th class='l'>ledger segment</th><th>calls</th><th>billed</th></tr></thead><tbody>" + srows +
               f"<tr class='tot'><td class='l'>total (seven reconciled segments)</td><td>{S['calls']:,}</td><td>${S['costUsd']:.5f}</td></tr></tbody></table>")

# ---- Table: in-browser feasibility (from the gates files) ---------------------
F = N["local_feasibility"]
order = ["Qwen3.5-2B", "Ministral-3-3B", "Qwen3-4B", "Phi-4-mini-instruct"]
vram = {"Qwen3.5-2B": 2245, "Ministral-3-3B": 2864, "Qwen3-4B": 3432, "Phi-4-mini-instruct": 3438}
def feas_row(prefix, label):
    out = []
    for m in order:
        k = next((k for k in F if k.startswith(f"{prefix}{m} [")), None)
        if not k: continue
        x = F[k]; nm = m.replace("-instruct", "")
        ans = f"{min(x['answered'], x['items'])}/{x['items']}"
        ttft = f"{x['ttft_p95']:,}" if x["ttft_p95"] is not None else "—"
        gate = "pass" if not x["killed_on_gates"] and x["answered"] else ("—" if not x["answered"] else "<b>killed</b>")
        out.append(f"<tr><td class='l'>{label} · {nm}</td><td>{vram[m]:,}</td><td>{x['engine_load_ms']:,}</td><td>{ans}</td><td>{x['budget_exhausted']}</td>"
                   f"<td>{ttft}</td><td>{x['item_wall_p50']:,}</td><td>{x['item_wall_p95']:,}</td><td>{x['decode_tok_s'] if x['decode_tok_s'] is not None else '—'}</td><td>{gate}</td></tr>")
    return "".join(out)
TABLE_FEAS = ("<table class='data'>{CAP}<thead><tr><th class='l'>configuration · model</th><th>VRAM MB</th><th>engine load ms</th><th>answered</th><th>budget-exhausted</th>"
              "<th>TTFT p95 ms</th><th>per-message p50 ms</th><th>p95 ms</th><th>decode tok/s</th><th>p95-TTFT gate</th></tr></thead><tbody>"
              + feas_row("tier2-", "compiled") + feas_row("baselineB-", "B") + "</tbody></table>")

# ---- Table: prevention (entity level) ------------------------------------------
Pv = N["prevention"]
import statistics as st, collections
prows = []
for m in order:
    k = next((k for k in Pv if k.startswith(f"tier2-{m} [")), None)
    if k: v = Pv[k]; prows.append(f"<tr><td class='l'>in-browser compiled · {m.replace('-instruct','')}</td><td>{v['fully_caught']}/{v['leak_bearing']}</td><td><b>{100*v['leak_prevention']:.0f}%</b></td><td>{v['over_blocked']}/{v['clean']}</td><td><b>{100*v['over_blocking']:.0f}%</b></td></tr>")
k = next(k for k in Pv if k.startswith("baselineB+tier0-Qwen3.5-2B ["))
v = Pv[k]; prows.append(f"<tr><td class='l'>in-browser B + tier 0 · Qwen3.5-2B</td><td>{v['fully_caught']}/{v['leak_bearing']}</td><td>{100*v['leak_prevention']:.0f}%</td><td>{v['over_blocked']}/{v['clean']}</td><td>{100*v['over_blocking']:.0f}%</td></tr>")
agg = collections.defaultdict(lambda: {"lp": [], "ob": []})
for k, v in Pv.items():
    if v["kind"] == "hosted" and "[ceiling-0" in k: a = k.split(" [")[0]; agg[a]["lp"].append(v["leak_prevention"]); agg[a]["ob"].append(v["over_blocking"])
for a, d in sorted(agg.items(), key=lambda kv: -st.mean(kv[1]["lp"])):
    prows.append(f"<tr><td class='l'>hosted B · {short(a).replace('B·','')} (3-pass mean)</td><td>—</td><td><b>{100*st.mean(d['lp']):.0f}%</b></td><td>—</td><td><b>{100*st.mean(d['ob']):.0f}%</b></td></tr>")
v = Pv["b-glm-5.3-flash [thinkonglm-01]"]
prows.append(f"<tr><td class='l'>hosted B · glm-5.3-flash, reasoning on (all rows; 59 unanswered)</td><td>{v['fully_caught']}/{v['leak_bearing']}</td><td>{100*v['leak_prevention']:.0f}%</td><td>{v['over_blocked']}/{v['clean']}</td><td>{100*v['over_blocking']:.0f}%</td></tr>")
TABLE_PREVENTION = ("<table class='data'>{CAP}<thead><tr><th class='l'>arm</th><th>leaks fully caught</th><th>prevention</th><th>clean messages actioned</th><th>over-blocking</th></tr></thead><tbody>"
                    + "".join(prows) + "</tbody></table>")

FIGS = {"FIG1": "f1-architecture.svg", "FIG_SCORE": "f2-scorecard.svg", "FIG_FEAS": "f3-feasibility.svg", "FIG_VAR": "f4-variance.svg",
        "FIG_THINK": "f5-thinking-on.svg", "FIG_LAT": "f6-latency.svg", "FIG_PR": "f8-pr-scatter.svg", "EFIG_SCORE": "e-scorecard.svg"}
def render(template, outname):
    out = open(os.path.join(HERE, template), encoding="utf8").read()
    tables = {"TABLE_CEILING": TABLE_CEILING, "TABLE_LATENCY": TABLE_LATENCY, "TABLE_SPEND": TABLE_SPEND, "TABLE_FEAS": TABLE_FEAS, "TABLE_PREVENTION": TABLE_PREVENTION}
    for k, tbl in tables.items():   # {{TABLE_X|CAP=...}} puts the caption INSIDE the table so it cannot be orphaned by a page break
        for m in list(re.finditer(r"\{\{" + k + r"\|CAP=(.*?)\}\}", out, re.S)):
            out = out.replace(m.group(0), tbl.replace("{CAP}", "<caption>" + m.group(1) + "</caption>"))
        out = out.replace("{{" + k + "}}", tbl.replace("{CAP}", ""))
    for k, f in FIGS.items():
        out = out.replace("{{" + k + "}}", fig(f))
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    assert not left, (outname, left)
    open(os.path.join(HERE, outname), "w", encoding="utf8").write(out); print("wrote", outname)
render("paper.template.html", "paper.html")
if os.path.exists(os.path.join(HERE, "executive-summary.template.html")): render("executive-summary.template.html", "executive-summary.html")
