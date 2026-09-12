#!/usr/bin/env python3
"""Aggregate runs into data/results/latest.json for the site. Effort = last run per effort bucket."""
import json, glob, os, datetime
BASE = os.path.dirname(os.path.abspath(__file__))
def runs_for(eff):
    """all runs across files whose reasoning == eff; 'default' counts as medium"""
    merged = {}
    for f in sorted(glob.glob(f"{BASE}/data/results/*_quick.json")):
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
    return out
res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "low": agg(runs_for("low")), "medium": agg(runs_for("medium")), "high": agg(runs_for("high"))}
json.dump(res, open(f"{BASE}/data/results/latest.json","w"), indent=1)
print("latest.json:", {k: len(v) for k,v in res.items() if k != "generated"})
