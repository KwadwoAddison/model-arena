#!/usr/bin/env python3
"""Aggregate runs into data/results/latest.json + per-model dossiers for the site.
Effort = newest run per effort bucket ('default' counts as medium).
Dossiers carry: per-task results, per-domain aggregates, grading rules, reasoning ladder,
composite score with the published weights, and the actual code_art SVG artifact."""
import json, glob, os, datetime
BASE = os.path.dirname(os.path.abspath(__file__))

ROSTER = {m["id"]: m for m in json.load(open(f"{BASE}/data/roster.json"))["models"]}
WEIGHTS = {"correctness": 0.40, "reasoning": 0.20, "token_eff": 0.15,
           "latency": 0.10, "adherence": 0.15}
# domain map fallback for old result files that lack the field
DOMAIN_OF = {"code_algo":"coding","code_pine":"coding","code_html":"coding","code_art":"coding",
             "code_b1":"coding","code_b2":"coding","code_b3":"coding",
             "med_neo":"medicine","med_surg":"medicine","med_pharm":"medicine","med_b1":"medicine","med_b2":"medicine",
             "trade_expect":"trading","trade_logic":"trading","trade_sizing":"trading","trade_b1":"trading","trade_b2":"trading",
             "long_privacy":"longform","long_pf":"longform","long_b1":"longform",
             "vision_b1":"vision","vision_b2":"vision"}
GRADING_RULES = {}
for f in sorted(glob.glob(f"{BASE}/data/results/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("grading_rules"): GRADING_RULES = d["grading_rules"]  # newest rules win

def runs_for(eff):
    """all runs across files whose reasoning == eff; 'default' counts as medium; full+quick merge"""
    merged = {}
    for f in sorted(glob.glob(f"{BASE}/data/results/*_quick.json")) + sorted(glob.glob(f"{BASE}/data/results/*_full.json")):
        try: d = json.load(open(f))
        except Exception: continue
        r_eff = d.get("reasoning", "default")
        if r_eff == "default": r_eff = "medium"
        if r_eff != eff: continue
        for r in d["runs"]:
            if "error" in r or r.get("correctness") is None: continue
            key = (r["model"], r["task"])
            merged[key] = r  # newest wins per model+task
    return list(merged.values())

def agg(rows):
    out = {}
    for r in rows:
        a = out.setdefault(r["model"], {"n":0,"corr":0.0,"lat":0.0,"tok":0.0})
        a["n"] += 1; a["corr"] += r["correctness"]; a["lat"] += r.get("latency_s",0); a["tok"] += r.get("tokens_est",0)
    for m,a in out.items():
        a["lat"] = round(a["lat"]/a["n"],1); a["tok"] = round(a["tok"]/a["n"]); a["corr"] = round(a["corr"],2)
        a["pct"] = round(100*a["corr"]/a["n"]) if a["n"] else 0
    return out

def domain_breakdown(rows):
    """per model per domain: pct + per-task list"""
    per = {}
    for r in rows:
        m, t = r["model"], r["task"]
        d = r.get("domain") or DOMAIN_OF.get(t, "other")
        pm = per.setdefault(m, {})
        pd = pm.setdefault(d, {"n":0,"corr":0.0,"tasks":{}})
        pd["n"] += 1; pd["corr"] += r["correctness"]
        pd["tasks"][t] = {"corr": r["correctness"], "lat": r.get("latency_s"),
                          "tok": r.get("tokens_est"), "chars": len(r.get("text",""))}
    for m in per:
        for d in per[m]:
            pd = per[m][d]
            pd["pct"] = round(100*pd["corr"]/pd["n"]) if pd["n"] else 0
            pd["corr"] = round(pd["corr"],2)
    return per

def best_worst(per):
    """strongest / weakest domain per model (min 1 task)"""
    bw = {}
    for m, doms in per.items():
        ranked = sorted(((d, v["pct"], v["n"]) for d, v in doms.items()), key=lambda x: (-x[1], -x[2]))
        if ranked:
            bw[m] = {"strongest": ranked[0][0], "weakest": ranked[-1][0],
                     "strongest_pct": ranked[0][1], "weakest_pct": ranked[-1][1]}
    return bw

def composite(med, high, low, lat_all):
    """published formula: correctness 40% (medium pct), reasoning 20% (0.6*high + 0.4*low pct),
    token_eff 15% (vs FAMILY MEDIAN tokens on correct runs — self-normalizing par),
    latency 10% (45s cap), adherence 15% (longform instruction-following pct)"""
    # family median tokens among correct medium runs = the par everyone is measured against
    import statistics
    pars = [v["tok"] for v in med.values() if v.get("pct", 0) >= 50 and v.get("tok")]
    par = statistics.median(pars) if pars else 1400
    comp = {}
    for m in med:
        medc = med[m].get("pct", 0)
        lo = low.get(m, {}).get("pct"); hi = high.get(m, {}).get("pct")
        r_qual = round(0.6*(hi if hi is not None else medc) + 0.4*(lo if lo is not None else medc))
        tok = med[m].get("tok", 9999)
        # tokens used / par: 1.0 at par, scales down for wasteful, floors at 0 (5x par = 0)
        t_eff = max(0, min(100, round(100*min(1.0, par/max(tok, par)))))
        lat = med[m].get("lat", 60)
        l_sc = max(0, min(100, round(100*(1 - min(1, lat/45)))))
        # adherence: longform domain pct (pure instruction-following), fall back to medium pct
        adm = per_dom.get(m, {}).get("longform", {}).get("pct", medc)
        comp[m] = {"correctness": medc, "reasoning": r_qual, "token_eff": t_eff, "latency": l_sc, "adherence": adm}
        comp[m]["total"] = round(sum(comp[m][k]*w for k, w in WEIGHTS.items() if k in comp[m]), 1)
    return comp

def art_artifacts(rows):
    """best code_art submission per model (for dossier showcase)"""
    arts = {}
    for r in rows:
        if r["task"] != "code_art": continue
        if "error" in r: continue
        cur = arts.get(r["model"])
        if cur is None or r["correctness"] > cur["corr"]:
            arts[r["model"]] = {"corr": r["correctness"], "svg": r.get("text","")[:6000],
                                "tok": r.get("tokens_est"), "lat": r.get("latency_s")}
    return arts

med_rows, high_rows, low_rows = runs_for("medium"), runs_for("high"), runs_for("low")
med, high, low = agg(med_rows), agg(high_rows), agg(low_rows)
per_dom = domain_breakdown(med_rows + high_rows + low_rows)
bw = best_worst(per_dom)
comp = composite(med, high, low, med)
arts = art_artifacts(med_rows + high_rows)

import statistics as _st
_pars = [v.get("tok") for v in med.values() if v.get("pct", 0) >= 50 and v.get("tok")]
res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "weights": WEIGHTS, "grading_rules": GRADING_RULES, "par": (_st.median(_pars) if _pars else None),
       "low": low, "medium": med, "high": high,
       "domains": per_dom, "best_worst": bw, "composite": comp, "art": arts}
json.dump(res, open(f"{BASE}/data/results/latest.json", "w"), indent=1)

# raw outputs per model/task (first 700 chars each) for dossier viewing
raw = {}
for r in (med_rows + high_rows + low_rows):
    if "error" in r: continue
    raw.setdefault(r["model"], {})[r["task"]] = {"grade": r.get("correctness"),
        "text": (r.get("text") or "")[:700], "lat": r.get("latency_s"), "tok": r.get("tokens_est")}
json.dump(raw, open(f"{BASE}/data/results/raw_latest.json", "w"), indent=1)
print("raw_latest.json:", sum(len(v) for v in raw.values()), "task outputs")
print("latest.json:", {k: len(v) for k,v in res.items() if isinstance(v, dict)})
print("models:", list(med))
for m in sorted(med):
    print(f"  {m:22} total={comp[m]['total']:5} corr={comp[m]['correctness']:3} reasoning={comp[m]['reasoning']:3} "
          f"tok_eff={comp[m]['token_eff']:3} lat_sc={comp[m]['latency']:3} strongest={bw.get(m,{}).get('strongest')} weakest={bw.get(m,{}).get('weakest')}")