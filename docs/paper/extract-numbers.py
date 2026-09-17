#!/usr/bin/env python3
"""Every number the paper prints, re-derived from the artifacts under runs/ and corpora/.

Pure Python, no dependencies. Writes docs/paper/data/numbers.json. Span-wise
predicate and span-level entity figures are parsed from the scorer's own output
(the canonical source); message-level figures, variance, latency, spend and the
thinking-ON comparison are computed here, using matching rules that were verified
to reproduce the scorer exactly (see the ceiling-arm record, Sec 7.5 and 10.7).
"""
import json, glob, os, re, collections, statistics as st, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs")
GOLD = os.path.join(ROOT, "corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl")
OUT = os.path.join(ROOT, "docs/paper/data/numbers.json")
CAP = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")

def rows(path): return [json.loads(l) for l in open(path, encoding="utf8")]
def prf(tp, fp, fn):
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return {"P": round(P,3), "R": round(R,3), "F1": round(F,3), "F1_raw": F, "tp": tp, "fp": fp, "fn": fn}
def pct(v, q):
    if not v: return None
    v = sorted(v); return v[min(len(v)-1, int(q*len(v)))]
def armkey(path):
    b = os.path.basename(path).replace(".jsonl", "")
    run, arm = b.split(".", 1)
    return run, arm.replace("ceiling-", "")

# ---- gold -------------------------------------------------------------------
gold = {}
for r in rows(GOLD):
    if r["status"] == "scored": gold[r["itemId"]] = r["satisfies"] is True
G = {"scored": len(gold), "positives": sum(gold.values()), "disputed": sum(1 for r in rows(GOLD) if r["status"] != "scored")}

def message_level(recs, answered_only=False):
    tp=fp=fn=0; ids=[]
    for r in recs:
        if r["itemId"] not in gold: continue
        if answered_only and not r.get("calls"): continue
        ids.append(r["itemId"])
        p = any(str(x.get("entityType","")).startswith("pred:") for x in (r.get("findings") or []))
        g = gold[r["itemId"]]
        if p and g: tp+=1
        elif p and not g: fp+=1
        elif not p and g: fn+=1
    return prf(tp,fp,fn), ids
def floor_message_level(texts, ids):
    tp=fp=fn=0
    for i in ids:
        h = bool(CAP.search(texts[i])); g = gold[i]
        if h and g: tp+=1
        elif h and not g: fp+=1
        elif not h and g: fn+=1
    return prf(tp,fp,fn)

files = sorted(glob.glob(os.path.join(RUNS, "ceiling-0[123].*.jsonl")))
texts = {r["itemId"]: r["text"] for r in rows(files[0])}
floor_all = floor_message_level(texts, list(gold))

# ---- message-level, every ceiling arm, every pass ----------------------------
msg = collections.defaultdict(dict)
unanswered = {}
for f in files + sorted(glob.glob(os.path.join(RUNS, "glmon-01.*.jsonl"))) + sorted(glob.glob(os.path.join(RUNS, "thinkonglm-01.*.jsonl"))):
    run, arm = armkey(f); recs = rows(f)
    s, _ = message_level(recs)
    msg[arm][run] = s
    dead = [r for r in recs if not r.get("calls") and r.get("error")]
    unanswered[f"{arm} [{run}]"] = {"unanswered": len(dead), "positives_lost": sum(1 for r in dead if gold.get(r["itemId"]))}

# ---- variance across passes 1-3 ----------------------------------------------
spreads = {}
for arm, byrun in msg.items():
    v = [byrun[f"ceiling-0{p}"]["F1"] for p in (1,2,3) if f"ceiling-0{p}" in byrun]
    raw = [byrun[f"ceiling-0{p}"]["F1_raw"] for p in (1,2,3) if f"ceiling-0{p}" in byrun]
    if len(v) == 3: spreads[arm] = {"passes": v, "spread": float(f"{max(raw)-min(raw):.3f}")}
msg_spread = [max(x["passes"])-min(x["passes"]) for x in spreads.values()]
msg_spread = [float(f"{max([byrun[f'ceiling-0{p}']['F1_raw'] for p in (1,2,3)])-min([byrun[f'ceiling-0{p}']['F1_raw'] for p in (1,2,3)]):.6f}") for a,byrun in msg.items() if all(f"ceiling-0{p}" in byrun for p in (1,2,3))]  # unrounded spreads; the mean is formatted once, matching the record's Sec 9.3

# ---- local arms on v2 at message level -------------------------------------
# Local runs are selected by the corpus their GATES file RECORDS, never by filename: the
# slate-corpus-01* runs record the superseded v1 corpus and were once mistaken for v2.
def gates_corpus(run):
    g = os.path.join(RUNS, f"{run}.gates.jsonl")
    if not os.path.exists(g): return None
    for r in rows(g):
        c = (r.get("run") or {}).get("corpus") or r.get("corpus")
        if c: return os.path.basename(str(c))
    return None
local_msg = {}
for f in sorted(glob.glob(os.path.join(RUNS, "slate-*.jsonl"))):
    if "gates" in f: continue
    if gates_corpus(os.path.basename(f).split(".",1)[0]) != "injection-p-fin-v2.labelled.jsonl": continue
    run, arm = os.path.basename(f).replace(".jsonl","").split(".",1)
    arm = arm.replace("-q4f16_1-MLC","").replace("-Instruct-2512-BF16","")
    s, ids = message_level(rows(f))
    if ids: local_msg[f"{arm} [{run}]"] = s

local_best_message = max(local_msg.items(), key=lambda kv: kv[1]["F1"])
# ---- scorer output: span-wise predicate + span-level entity ------------------
SC = os.path.join(ROOT, "docs/paper/data/score-final.txt")
if not os.path.exists(SC):
    out = subprocess.run(["pnpm","-C",os.path.join(ROOT,"apps/eval"),"run","-s","ceiling:score"],capture_output=True,text=True).stdout
    open(SC,"w",encoding="utf8").write(out)
sc = open(SC, encoding="utf8").read().split("\n")
def section(start_pat, end_pat):
    i = next(k for k,l in enumerate(sc) if start_pat in l)
    j = next((k for k,l in enumerate(sc) if k>i and end_pat in l), len(sc))
    return sc[i:j]
pred = section("## PREDICATE level — gold-tier2-predicate", "## PREDICATE level — gold-tier2 (")
spanwise = {}
for l in pred:
    m = re.match(r"\| (.+?) \| (ceiling|local) \| ([\d.—]+) \| ([\d.—]+) \| ([\d.—]+) \|", l)
    if m: spanwise[m.group(1)] = {"kind": m.group(2), "P": m.group(3), "R": m.group(4), "F1": m.group(5)}
floors_spanwise = {}
for l in pred:
    m = re.match(r"\| \*\*FLOOR (.+?)\*\* \| floor \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
    if m: floors_spanwise[m.group(1)] = {"P": m.group(2), "R": m.group(3), "F1": m.group(4)}
span = section("## SPAN level", "## Transport")
spanlevel = {}; span_floors = {}
for l in span:
    m = re.match(r"\| (.+?) \| (ceiling|local) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", l)
    if m: spanlevel[m.group(1)] = {"kind": m.group(2), "findings": int(m.group(3)), "tp": int(m.group(4)), "fp": int(m.group(5)), "fn": int(m.group(6)), "P": m.group(7), "R": m.group(8), "F1": m.group(9)}
    m2 = re.match(r"FLOOR — orthographic oracle, (budget-matched|unbudgeted).*?: P ([\d.]+) R ([\d.]+) F1 ([\d.]+)", l)
    if m2: span_floors[m2.group(1)] = {"P": m2.group(2), "R": m2.group(3), "F1": m2.group(4)}
attempted = []
for l in section("### ATTEMPTED-ONLY", "arms whose verdict"):
    m = re.match(r"\| (.+?) \| (ceiling|local) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([+\-][\d.]+) \|", l)
    if m: attempted.append({"arm": m.group(1), "unanswered": int(m.group(3)), "positives_lost": int(m.group(4)), "whole_F1": m.group(5), "whole_floor": m.group(6), "att_F1": m.group(7), "att_floor": m.group(8)})

# ---- latency, pooled over passes 1-3 ----------------------------------------
lat = collections.defaultdict(lambda: {"ttft":[], "cw":[], "iw_ans":[], "ct":[], "n429":0, "calls":0, "rt":[]})
for f in files:
    _, arm = armkey(f); a = lat[arm]
    for r in rows(f):
        if r.get("calls"): a["iw_ans"].append(r["wallMs"])
        for c in r.get("calls") or []:
            a["calls"]+=1
            if c.get("ttftMs") is not None: a["ttft"].append(c["ttftMs"])
            a["cw"].append(c["wallMs"])
            if c.get("completionTokens") is not None: a["ct"].append(c["completionTokens"])
            if c.get("reasoningTokens") is not None: a["rt"].append(c["reasoningTokens"])
            a["n429"] += sum(1 for x in (c.get("retries") or []) if x.get("status")==429)
latency = {arm: {"calls": a["calls"], "ttft_p50": round(pct(a["ttft"],.5)), "ttft_p95": round(pct(a["ttft"],.95)),
                 "call_wall_p50": round(pct(a["cw"],.5)), "item_wall_p50_answered": round(pct(a["iw_ans"],.5)),
                 "completion_p50": round(pct(a["ct"],.5)), "n429": a["n429"],
                 "reasoning_zero": sum(1 for x in a["rt"] if x==0), "reasoning_n": len(a["rt"])} for arm,a in lat.items()}

# ---- structured output & reasoning, thinking-off, passes 1-3 -----------------
so = collections.Counter(); tot=0; rz=0; rn=0; over512=0; trunc=0
for f in files:
    for r in rows(f):
        for c in r.get("calls") or []:
            tot+=1; so[str(c.get("parse"))]+=1
            if c.get("reasoningTokens") == 0: rz+=1
            if c.get("reasoningTokens") is None: rn+=1
            if (c.get("completionTokens") or 0) >= 512: over512+=1
            if c.get("finishReason")=="length": trunc+=1
structured = {"calls": tot, "parse": dict(so), "reasoning_exactly_zero": rz, "reasoning_null": rn, "completion_ge_512": over512, "truncated": trunc}

# ---- spend -------------------------------------------------------------------
segs = [("probe","ceiling-probe.spend.json"),("ceiling-01 window 1","ceiling-ceiling-01.part1.spend.json"),("glmon-01","ceiling-glmon-01.spend.json"),
        ("ceiling-01 window 2","ceiling-ceiling-01.spend.json"),("ceiling-02+03","ceiling-ceiling-02.spend.json"),
        ("thinkon-01 (aborted)","orphaned/thinkon-fullslate-aborted.spend.json"),("thinkonglm-01","ceiling-thinkonglm-01.spend.json")]
spend = []; bym = collections.Counter(); byf = collections.Counter(); est=0.0
for name, fn in segs:
    d = json.load(open(os.path.join(RUNS, fn)))
    spend.append({"segment": name, "calls": d["calls"], "costUsd": round(d["costUsd"],5)})
    est += d["estimatedCostUsd"]
    for k,v in d["byModel"].items(): bym[k.split("/")[-1]] += v
    for k,v in d["byFamily"].items(): byf[k] += v
toff = [json.load(open(os.path.join(RUNS, fn))) for i, (_, fn) in enumerate(segs) if i in (0, 1, 3, 4)]   # thinking-off only: glmon-01 is a thinking-ON run
ton = [json.load(open(os.path.join(RUNS, fn))) for i, (_, fn) in enumerate(segs) if i in (2, 5, 6)]
tbf = collections.Counter()
for d in toff:
    for k, v in d["byFamily"].items(): tbf[k] += v
thinkoff = {"calls": sum(d["calls"] for d in toff), "costUsd": round(sum(d["costUsd"] for d in toff),6), "estimateUsd": round(sum(d["estimatedCostUsd"] for d in toff),6), "byFamily": {k: round(v,5) for k,v in tbf.items()}}
thinkon_sub = {"calls": sum(d["calls"] for d in ton), "costUsd": round(sum(d["costUsd"] for d in ton),6)}
spend_total = {"thinkoff": thinkoff, "thinkon": thinkon_sub, "calls": sum(s["calls"] for s in spend), "costUsd": round(sum(s["costUsd"] for s in spend),6), "estimateUsd": round(est,6),
               "byModel": {k: round(v,5) for k,v in bym.most_common()}, "byFamily": {k: round(v,5) for k,v in byf.items()},
               "keyFinalUsd": 1.29213, "keyLimitUsd": 10, "hardStopUsd": 7}

# ---- thinking-ON: GLM 600 vs 8192 -------------------------------------------
def glm_block(path):
    recs = rows(path); ans = [r for r in recs if r.get("calls")]
    s_all,_ = message_level(recs); s_att, ids = message_level(recs, answered_only=True)
    fl = floor_message_level(texts, ids)
    tr = sum(1 for r in recs if any(c.get("finishReason")=="length" for c in r["calls"] or []))
    hit_c=hit_t=miss_c=miss_t=0
    for r in recs:
        if not gold.get(r["itemId"]): continue
        cs = r.get("calls") or []
        if not cs: continue
        t = any(c.get("finishReason")=="length" for c in cs)
        p = any(str(x.get("entityType","")).startswith("pred:") for x in (r.get("findings") or []))
        if p: (hit_t if t else hit_c)  # noqa
        if p and t: hit_t+=1
        elif p: hit_c+=1
        elif t: miss_t+=1
        else: miss_c+=1
    rt = sorted(c["reasoningTokens"] for r in recs for c in (r.get("calls") or []) if c.get("reasoningTokens") is not None)
    return {"rows": len(recs), "answered": len(ans), "scored_answered": len(ids), "positives_answered": sum(1 for i in ids if gold[i]),
            "truncated_rows": tr, "reasoning_p50": rt[len(rt)//2] if rt else None, "reasoning_max": max(rt) if rt else None,
            "message_whole": s_all, "message_attempted": s_att, "floor_attempted": fl,
            "split": {"detected_clean": hit_c, "detected_truncated": hit_t, "missed_clean": miss_c, "missed_truncated": miss_t}}
thinkon = {"glm_600": glm_block(os.path.join(RUNS,"glmon-01.ceiling-judge-glm-5.3-flash.jsonl")),
           "glm_8192": glm_block(os.path.join(RUNS,"thinkonglm-01.ceiling-judge-glm-5.3-flash.jsonl"))}
# B arm entity spans, overlap rule (verified to reproduce the scorer), whole and answered
def entity_overlap(recs):
    tp=fp=fn=0
    for r in recs:
        gs=[(x["start"],x["end"]) for x in (r.get("gold") or [])]
        fs=[(x["start"],x["end"]) for x in (r.get("findings") or []) if not str(x.get("entityType","")).startswith("pred:")]
        used=set(); hit=0
        for a in fs:
            for j,b in enumerate(gs):
                if j in used: continue
                if a[0]<b[1] and b[0]<a[1]: used.add(j); hit+=1; break
        tp+=hit; fp+=len(fs)-hit; fn+=len(gs)-hit
    return prf(tp,fp,fn)
brecs = rows(os.path.join(RUNS,"thinkonglm-01.ceiling-b-glm-5.3-flash.jsonl"))
thinkon["glm_b_8192"] = {"answered": sum(1 for r in brecs if r.get("calls")), "entity_whole": entity_overlap(brecs),
                         "entity_answered": entity_overlap([r for r in brecs if r.get("calls")]),
                         "truncated_rows": sum(1 for r in brecs if any(c.get("finishReason")=="length" for c in r.get("calls") or []))}
# oracle on the answered subset, via the REAL implementation
helper = os.path.join(ROOT, "docs/paper/oracle-subset.ts")
res = subprocess.run(["pnpm","-C",os.path.join(ROOT,"apps/eval"),"exec","vite-node","--config","vite-node.config.ts",helper,
                      os.path.join(RUNS,"thinkonglm-01.ceiling-b-glm-5.3-flash.jsonl")],capture_output=True,text=True)
oracle = {}
for l in res.stdout.split("\n"):
    m = re.match(r"(ALL|ANSWERED) (\d+): budget-matched F1 ([\d.]+) .*unbudgeted F1 ([\d.]+)", l)
    if m: oracle[m.group(1).lower()] = {"rows": int(m.group(2)), "budget_matched_F1": float(m.group(3)), "unbudgeted_F1": float(m.group(4))}
if not oracle: print("WARN oracle helper produced nothing:", res.stderr[-400:], file=sys.stderr)
thinkon["oracle_on_glm_b_rows"] = oracle

# ---- in-browser feasibility, from the gates files + per-item wall time -------
def q(v, p):
    if not v: return None
    v = sorted(v); return v[min(len(v)-1, int(p*len(v)))]
local_feas = {}
for g in sorted(glob.glob(os.path.join(RUNS, "slate-rebuild-*.gates.jsonl"))):
    run = os.path.basename(g).replace(".gates.jsonl", "")
    for r in rows(g):
        arm = r["arm"].replace("-q4f16_1-MLC", "").replace("-Instruct-2512-BF16", "")
        recs = rows(os.path.join(RUNS, f"{run}.{r['arm']}.jsonl"))
        t2 = [x["timings"]["tier2Ms"] for x in recs if x.get("timings") and x["timings"].get("tier2Ms") is not None]
        local_feas[f"{arm} [{run}]"] = {
            "family": r["family"], "items": r["items"], "answered": r["answeredCalls"],
            "budget_exhausted": (r.get("degradedItems") or {}).get("budget-exhausted", 0),
            "ttft_p95": round(r["ttftMs"]["p95"]) if r.get("ttftMs") and r["ttftMs"].get("p95") is not None else None,
            "prompt_p50": (r.get("promptTokens") or {}).get("p50"), "completion_p50": (r.get("completionTokens") or {}).get("p50"),
            "decode_tok_s": round(r["sustainedDecodeTokPerSec"], 1) if r.get("sustainedDecodeTokPerSec") else None,
            "engine_load_ms": round(r.get("engineLoadMs") or 0), "engine_warmup_ms": round(r.get("engineWarmupMs") or 0),
            "budget_ms": r["run"]["latencyBudgetMs"], "killed_on_gates": r.get("killedOnRunGates"),
            "item_wall_p50": round(q(t2, .5)) if t2 else None, "item_wall_p95": round(q(t2, .95)) if t2 else None,
        }

# answered-only per-message median for the in-browser arms (rows not budget-exhausted)
for k, v in local_feas.items():
    arm, run = k.split(" [")[0], k.split(" [")[1].rstrip("]")
    raw = next(a for a in os.listdir(RUNS) if a.startswith(run + ".") and a.replace("-q4f16_1-MLC", "").replace("-Instruct-2512-BF16", "") == f"{run}.{arm}.jsonl")
    recs = rows(os.path.join(RUNS, raw))
    ans = [x["timings"]["tier2Ms"] for x in recs if x.get("timings") and not any(d.get("reason") == "budget-exhausted" for d in (x.get("degraded") or []))]
    v["answered_wall_p50"] = round(q(ans, .5)) if ans else None
# span-ladder health of the in-browser arms, from the gates
local_ladder = {}
for g in sorted(glob.glob(os.path.join(RUNS, "slate-rebuild-*.gates.jsonl"))):
    run = os.path.basename(g).replace(".gates.jsonl", "")
    for r in rows(g):
        arm = r["arm"].replace("-q4f16_1-MLC", "").replace("-Instruct-2512-BF16", "")
        lad = r.get("ladder") or {}
        rr = next((x for x in (r.get("gates") or []) if x.get("gate") == "resolvable-rate"), {})
        local_ladder[f"{arm} [{run}]"] = {"resolved": (lad.get("rung1") or 0) + (lad.get("rung2") or 0),
                                          "unresolvedQuotes": lad.get("unresolvedQuotes"), "unresolvedMentions": lad.get("unresolvedMentions"), "unknownLabels": lad.get("unknownLabels"),
                                          "resolvable_rate": (round(rr["observed"], 3) if rr.get("observed") is not None else None), "resolvable_floor": rr.get("threshold"), "verdict": rr.get("verdict"),
                                          "killed": [x["gate"] for x in (r.get("gates") or []) if x.get("verdict") == "fail"]}

# ---- leak-prevention / over-blocking at the entity level (spec 6.4 metric 1) ----
def prevention(recs):
    bearing = caught = clean = blocked = 0
    for r in recs:
        gs = [(x["start"], x["end"]) for x in (r.get("gold") or []) if not str(x.get("entityType", "")).startswith("pred:")]
        fs = [(x["start"], x["end"]) for x in (r.get("findings") or []) if not str(x.get("entityType", "")).startswith("pred:")]
        if gs:
            bearing += 1
            if all(any(f[0] < g[1] and g[0] < f[1] for f in fs) for g in gs): caught += 1
        else:
            clean += 1
            if fs: blocked += 1
    return {"leak_bearing": bearing, "fully_caught": caught, "leak_prevention": round(caught/bearing, 3) if bearing else None,
            "clean": clean, "over_blocked": blocked, "over_blocking": round(blocked/clean, 3) if clean else None}
prev = {}
for g in sorted(glob.glob(os.path.join(RUNS, "slate-rebuild-*.gates.jsonl"))):
    run = os.path.basename(g).replace(".gates.jsonl", "")
    for r in rows(g):
        arm = r["arm"].replace("-q4f16_1-MLC", "").replace("-Instruct-2512-BF16", "")
        prev[f"{arm} [{run}]"] = {"kind": "in-browser", **prevention(rows(os.path.join(RUNS, f"{run}.{r['arm']}.jsonl")))}
for f in sorted(glob.glob(os.path.join(RUNS, "ceiling-0[123].ceiling-b-*.jsonl"))) + [os.path.join(RUNS, "thinkonglm-01.ceiling-b-glm-5.3-flash.jsonl")]:
    run, arm = armkey(f); prev[f"{arm} [{run}]"] = {"kind": "hosted", **prevention(rows(f))}

# ---- cost per 1,000 messages ---------------------------------------------------
per_model_msgs = 3 * 189 * 2   # three thinking-off passes, two families
cost_1k = {"thinkoff_per_model": {m.split("/")[-1]: round(v / per_model_msgs * 1000, 4) for m, v in
           collections.Counter({k: v for d in toff for k, v in d["byModel"].items()}).items()},
           "thinkoff_judge": round(tbf["judge"] / (5 * 3 * 189) * 1000, 4), "thinkoff_b": round(tbf["b"] / (5 * 3 * 189) * 1000, 4)}
# byModel across the five thinking-off segments, summed properly
_bm = collections.Counter()
for d in toff:
    for k, v in d["byModel"].items(): _bm[k.split("/")[-1]] += v
cost_1k["thinkoff_per_model"] = {k: round(v / per_model_msgs * 1000, 4) for k, v in _bm.items() if k != "glm-5.3-flash"}
_g = json.load(open(os.path.join(RUNS, "ceiling-thinkonglm-01.spend.json")))["byFamily"]
cost_1k["thinkon_glm_judge"] = round(_g["judge"] / 189 * 1000, 4); cost_1k["thinkon_glm_b"] = round(_g["b"] / 189 * 1000, 4)

# ---- corpus facts -------------------------------------------------------------
lab = rows(os.path.join(ROOT,"corpora/generated/injection-p-fin-v2.labelled.jsonl"))
ents = [g for r in lab for g in (r.get("gold") or r.get("spans") or [])]
corpus = {"items": len(lab), "entity_spans": len(ents), "entity_types": sorted({g.get("entityType") for g in ents if g.get("entityType")}),
          "items_with_entity": sum(1 for r in lab if (r.get("gold") or r.get("spans"))), "predicate_gold": G}

ent_spread = {}
for a in ("b-nemotron-3-super-120b-a12b","b-deepseek-v4-flash-0731","b-qwen3.8-27b","b-mistral-small-2603","b-qwen3.8-flash"):
    v = [float(spanlevel[f"ceiling-{a} [ceiling-0{p}]"]["F1"]) for p in (1,2,3)]
    ent_spread[a] = float(f"{max(v)-min(v):.3f}")
# ---- the TypeSafe judgment arm (jev-1.13.0), three passes -------------------
TS_PASSES = ["ts-01", "ts-02", "ts-03"]
ts = None
ts_scores = None
if all(os.path.exists(os.path.join(RUNS, f"{p}.ts-judgment.score.json")) for p in TS_PASSES):
    S = [json.load(open(os.path.join(RUNS, f"{p}.ts-judgment.score.json"))) for p in TS_PASSES]
    gates = [json.loads(open(os.path.join(RUNS, f"{p}.ts-judgment.gates.jsonl")).readline()) for p in TS_PASSES]
    POS = S[0]["gold"]["positives"]; SCORED = S[0]["gold"]["scored"]; NEG = SCORED - POS
    def point(d):
        return {"P": round(d["precision"], 3), "R": round(d["recall"], 3), "F1": round(d["f1"], 3),
                "tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": NEG - d["fp"],
                "fpr": round(d["fp"] / NEG, 4), "fnr": round(d["fn"] / POS, 4)}
    def ent(d):
        return {"F1": round(d["f1"], 3), "P": round(d["precision"], 3), "R": round(d["recall"], 3),
                "leak_prevention": round(d["leakPrevention"], 3), "over_blocking": round(d["overBlocking"], 3),
                "fully_caught": d["fullyCaught"], "leak_bearing": d["leakBearing"], "over_blocked": d["overBlocked"], "clean": d["clean"],
                "tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "threshold": round(d["threshold"], 2)}
    def at(sweep, t):
        return next(x for x in sweep if abs(x["threshold"] - t) < 1e-9)
    f1s = [s["predicate"]["at0_5"]["f1"] for s in S]
    ts = {
        "passes": TS_PASSES, "model": gates[0]["modelReturned"], "requested_model": gates[0]["model"],
        "gold": {"scored": SCORED, "positives": POS, "negatives": NEG},
        "predicate_at_half": {p: point(s["predicate"]["at0_5"]) for p, s in zip(TS_PASSES, S)},
        "predicate_best": {p: {"threshold": round(s["predicate"]["best"]["threshold"], 3), "F1": round(s["predicate"]["best"]["f1"], 3),
                               "fp": s["predicate"]["best"]["fp"], "fn": s["predicate"]["best"]["fn"]} for p, s in zip(TS_PASSES, S)},
        "split_half": {p: round(s["predicate"]["splitHalf"]["mean"], 3) for p, s in zip(TS_PASSES, S)},
        "roc_auc": {p: round(s["predicate"]["rocAuc"], 4) for p, s in zip(TS_PASSES, S)},
        "average_precision": {p: round(s["predicate"]["averagePrecision"], 4) for p, s in zip(TS_PASSES, S)},
        "mean": {"F1_at_half": round(st.mean(f1s), 3), "split_half": round(st.mean(s["predicate"]["splitHalf"]["mean"] for s in S), 3),
                 "roc_auc": round(st.mean(s["predicate"]["rocAuc"] for s in S), 4),
                 "fpr": round(st.mean(s["predicate"]["at0_5"]["fp"] for s in S) / NEG, 4),
                 "F1_spread": round(max(f1s) - min(f1s), 3)},
        "entity_at_half": {p: ent(s["entity"]["at0_5"]) for p, s in zip(TS_PASSES, S)},
        "entity_at_85": {p: ent(at(s["entity"]["sweep"], 0.85)) for p, s in zip(TS_PASSES, S)},
        "entity_at_95": {p: ent(at(s["entity"]["sweep"], 0.95)) for p, s in zip(TS_PASSES, S)},
        "entity_tier0": ent(S[0]["entity"]["tier0Baseline"]),
        "entity_sweep": [{"threshold": round(x["threshold"], 2), "leak_prevention": round(x["leakPrevention"], 3),
                          "over_blocking": round(x["overBlocking"], 3), "F1": round(x["f1"], 3)} for x in S[0]["entity"]["sweep"]],
        "filter": {p: {"candidates": s["filterEffect"]["tier0Candidates"], "on_gold": s["filterEffect"]["onGold"], "off_gold": s["filterEffect"]["offGold"],
                       "correct_rejections": s["filterEffect"]["correctRejections"], "wrongful_rejections": s["filterEffect"]["wrongfulRejections"],
                       "rejection_rate_off_gold": round(s["filterEffect"]["rejectionRateOffGold"], 3),
                       "rejection_rate_on_gold": round(s["filterEffect"]["rejectionRateOnGold"], 3)} for p, s in zip(TS_PASSES, S)},
        "cost_time": {"per_1k_usd": round(st.mean(s["costAndTime"]["costPer1kMessages"] for s in S), 4),
                      "cost_per_pass_usd": round(st.mean(s["costAndTime"]["costUsd"] for s in S), 6),
                      "item_wall_p50": round(st.mean(s["costAndTime"]["itemWallMsP50"] for s in S)),
                      "item_wall_p95": round(st.mean(s["costAndTime"]["itemWallMsP95"] for s in S)),
                      "input_tokens_p50": round(st.mean(s["costAndTime"]["inputTokensP50"] for s in S)),
                      "questions_p50": round(st.mean(s["costAndTime"]["questionsPerItemP50"] for s in S))},
        "ops": {"items": gates[0]["items"], "answered": [g["answered"] for g in gates], "errored": [g["errored"] for g in gates],
                "retries": [g["retries"] for g in gates], "rate_limited": [g["rateLimited"] for g in gates]},
        "calibration": [{"lower": b["lower"], "n": b["n"], "mean_p": None if b["meanProbability"] is None else round(b["meanProbability"], 3),
                         "observed": None if b["observedRate"] is None else round(b["observedRate"], 3)} for b in S[0]["predicate"]["calibration"] if b["n"] > 0],
    }
    # The separation strip: every scored message's probability, split by gold label.
    # Read from the ARM file and the gold, not from the score file, so the figure
    # cannot show a distribution the scorer smoothed.
    tsgold = {}
    for r in rows(os.path.join(ROOT, "corpora/generated/injection-p-fin-v2.gold-tier2-predicate.jsonl")):
        if r.get("status") == "scored" and isinstance(r.get("satisfies"), bool):
            tsgold[r["itemId"]] = r["satisfies"]
    pos, neg = [], []
    for r in rows(os.path.join(RUNS, "ts-01.ts-judgment.jsonl")):
        lab = tsgold.get(r["itemId"])
        if lab is None or r.get("predicateProbability") is None: continue
        (pos if lab else neg).append(round(float(r["predicateProbability"]), 4))
    ts_scores = {"pass": TS_PASSES[0], "positives": sorted(pos), "negatives": sorted(neg),
                 "highest_negative": max(neg), "lowest_positive": min(pos),
                 "gap": round(min(pos) - max(neg), 4)}
    # Cross-pass agreement, which is the stability claim.
    recs = {p: {r["itemId"]: r for r in rows(os.path.join(RUNS, f"{p}.ts-judgment.jsonl"))} for p in TS_PASSES}
    ids = sorted(recs[TS_PASSES[0]])
    trip = [[recs[p][i]["predicateProbability"] for p in TS_PASSES] for i in ids]
    labs = [[recs[p][i]["candidates"][k]["choice"] for p in TS_PASSES] for i in ids for k in range(len(recs[TS_PASSES[0]][i]["candidates"]))]
    ts["stability"] = {"identical_predicate": sum(1 for t in trip if t[0] == t[1] == t[2]), "items": len(trip),
                       "max_probability_spread": round(max(max(t) - min(t) for t in trip), 4),
                       "identical_labels": sum(1 for l in labs if l[0] == l[1] == l[2]), "candidates": len(labs)}

out = {"gold": G, "corpus": corpus, "floor_message_level_all": floor_all, "message_level": msg, "unanswered": unanswered,
       "local_best_message": {"arm": local_best_message[0], **local_best_message[1]},
       "variance": {"per_arm": spreads, "mean_spread": float(f"{st.mean(msg_spread):.3f}"), "max_spread": float(f"{max(msg_spread):.3f}"),
                    "entity_per_arm": ent_spread, "entity_mean_spread": float(f"{st.mean(ent_spread.values()):.3f}"), "entity_max_spread": float(f"{max(ent_spread.values()):.3f}")},
       "local_message_level": local_msg, "spanwise_predicate": spanwise, "floors_spanwise": floors_spanwise,
       "spanlevel_entity": spanlevel, "span_floors": span_floors, "attempted_only": attempted, "latency": latency,
       "structured": structured, "spend": {"segments": spend, **spend_total}, "thinkon": thinkon,
       "local_feasibility": local_feas, "local_ladder": local_ladder, "prevention": prev, "cost_per_1k": cost_1k,
       "typesafe": ts, "typesafe_scores": ts_scores}
json.dump(out, open(OUT,"w",encoding="utf8"), indent=1)
print("wrote", OUT)
print(f"  gold {G}  floor(msg) {floor_all['F1']}  variance mean {out['variance']['mean_spread']} max {out['variance']['max_spread']}")
print(f"  spanwise rows {len(spanwise)}  spanlevel rows {len(spanlevel)}  attempted rows {len(attempted)}  latency arms {len(latency)}")
print(f"  structured {structured}")
print(f"  spend total {spend_total['calls']} calls ${spend_total['costUsd']}")
print(f"  thinkon 600 msg-att {thinkon['glm_600']['message_attempted']['F1']} vs floor {thinkon['glm_600']['floor_attempted']['F1']} | 8192 {thinkon['glm_8192']['message_attempted']['F1']} vs {thinkon['glm_8192']['floor_attempted']['F1']}")
if ts: print(f"  typesafe F1@0.5 {ts['mean']['F1_at_half']} split-half {ts['mean']['split_half']} AUC {ts['mean']['roc_auc']} FPR {ts['mean']['fpr']} | gap {ts_scores['gap']} | ${ts['cost_time']['per_1k_usd']}/1k {ts['cost_time']['item_wall_p50']}ms")
print(f"  glm-b entity whole {thinkon['glm_b_8192']['entity_whole']['F1']} answered {thinkon['glm_b_8192']['entity_answered']['F1']}  oracle {oracle}")
