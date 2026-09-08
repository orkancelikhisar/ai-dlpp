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
toff = [json.load(open(os.path.join(RUNS, fn))) for _, fn in segs[:5]]
tbf = collections.Counter()
for d in toff:
    for k, v in d["byFamily"].items(): tbf[k] += v
thinkoff = {"calls": sum(d["calls"] for d in toff), "costUsd": round(sum(d["costUsd"] for d in toff),6), "estimateUsd": round(sum(d["estimatedCostUsd"] for d in toff),6), "byFamily": {k: round(v,5) for k,v in tbf.items()}}
spend_total = {"thinkoff": thinkoff, "calls": sum(s["calls"] for s in spend), "costUsd": round(sum(s["costUsd"] for s in spend),6), "estimateUsd": round(est,6),
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

# ---- corpus facts -------------------------------------------------------------
lab = rows(os.path.join(ROOT,"corpora/generated/injection-p-fin-v2.labelled.jsonl"))
ents = [g for r in lab for g in (r.get("gold") or r.get("spans") or [])]
corpus = {"items": len(lab), "entity_spans": len(ents), "entity_types": sorted({g.get("entityType") for g in ents if g.get("entityType")}),
          "items_with_entity": sum(1 for r in lab if (r.get("gold") or r.get("spans"))), "predicate_gold": G}

ent_spread = {}
for a in ("b-nemotron-3-super-120b-a12b","b-deepseek-v4-flash-0731","b-qwen3.8-27b","b-mistral-small-2603","b-qwen3.8-flash"):
    v = [float(spanlevel[f"ceiling-{a} [ceiling-0{p}]"]["F1"]) for p in (1,2,3)]
    ent_spread[a] = float(f"{max(v)-min(v):.3f}")
out = {"gold": G, "corpus": corpus, "floor_message_level_all": floor_all, "message_level": msg, "unanswered": unanswered,
       "local_best_message": {"arm": local_best_message[0], **local_best_message[1]},
       "variance": {"per_arm": spreads, "mean_spread": float(f"{st.mean(msg_spread):.3f}"), "max_spread": float(f"{max(msg_spread):.3f}"),
                    "entity_per_arm": ent_spread, "entity_mean_spread": float(f"{st.mean(ent_spread.values()):.3f}"), "entity_max_spread": float(f"{max(ent_spread.values()):.3f}")},
       "local_message_level": local_msg, "spanwise_predicate": spanwise, "floors_spanwise": floors_spanwise,
       "spanlevel_entity": spanlevel, "span_floors": span_floors, "attempted_only": attempted, "latency": latency,
       "structured": structured, "spend": {"segments": spend, **spend_total}, "thinkon": thinkon}
json.dump(out, open(OUT,"w",encoding="utf8"), indent=1)
print("wrote", OUT)
print(f"  gold {G}  floor(msg) {floor_all['F1']}  variance mean {out['variance']['mean_spread']} max {out['variance']['max_spread']}")
print(f"  spanwise rows {len(spanwise)}  spanlevel rows {len(spanlevel)}  attempted rows {len(attempted)}  latency arms {len(latency)}")
print(f"  structured {structured}")
print(f"  spend total {spend_total['calls']} calls ${spend_total['costUsd']}")
print(f"  thinkon 600 msg-att {thinkon['glm_600']['message_attempted']['F1']} vs floor {thinkon['glm_600']['floor_attempted']['F1']} | 8192 {thinkon['glm_8192']['message_attempted']['F1']} vs {thinkon['glm_8192']['floor_attempted']['F1']}")
print(f"  glm-b entity whole {thinkon['glm_b_8192']['entity_whole']['F1']} answered {thinkon['glm_b_8192']['entity_answered']['F1']}  oracle {oracle}")
