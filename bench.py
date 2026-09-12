#!/usr/bin/env python3
"""
Benchmark harness — Dr. Addison's 7-model family.
Fair by design: same prompts, same grader, blind scoring, par-token efficiency,
reasoning-level sweep, JSON results only (site renders, it never scores).
Usage:
  python3 bench.py --suite quick            # smoke: 1 task/domain
  python3 bench.py --suite full             # all tasks
  python3 bench.py --task code_b1 --models glm-5.3,deepseek-v4.1-flash
  python3 bench.py --reasoning high --suite full
Results -> data/results/<timestamp>_<suite>.json  (never edited by hand)
"""
import argparse, datetime, json, os, re, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
ROSTER = json.load(open(f"{BASE}/data/roster.json"))
RESULTS = f"{BASE}/data/results"
os.makedirs(RESULTS, exist_ok=True)

# ---------- task suites (each task: id, domain, prompt, grader, par_tokens) ----------
TASKS = {
  # --- coding: deterministic grader ---
  "code_b1": dict(domain="coding", par_tokens=2200,
    prompt="Write a Python function `solve(s)` that returns the length of the longest substring with at most k distinct characters (k=2). Include only the function, no tests.",
    grade="unit"),
  "code_b2": dict(domain="coding", par_tokens=2600,
    prompt="Write a Pine Script v6 function `f_room(highs, lows, atr)` that returns the distance in ATRs from last close to the nearest swing high above or swing low below (lookback 20), as a float. Pine v6 syntax only.",
    grade="manual_heuristic"),
  "code_b3": dict(domain="coding", par_tokens=3000,
    prompt="Write a single-file HTML page (no external deps) that shows an animated count-up number 61 to 0 on scroll using IntersectionObserver, guaranteed to complete even if the tab is backgrounded during load. Output only the HTML.",
    grade="manual_heuristic"),
  # --- vision: needs image path; graded by checklist match ---
  "vision_b1": dict(domain="vision", par_tokens=1500,
    prompt="Read the attached chart image. Report: instrument, timeframe, and the last 3 swing highs/lows as numbers. Be exact.",
    image="data/images/chart_current.png", grade="checklist"),
  "vision_b2": dict(domain="vision", par_tokens=1200,
    prompt="Read the attached handwritten clinical note crop. Transcribe every number you can see (vitals, labs), labeled. Do not invent unreadable values; mark unclear ones.",
    image="data/images/note_crop.png", grade="checklist"),
  # --- medicine: curriculum MCQ with key ---
  "med_b1": dict(domain="medicine", par_tokens=1800,
    prompt="A 2-day-old term baby, APGAR 9/10, bilious vomiting, no anal patible fistula. Abdominal X-ray: double bubble, no distal gas. Next step? A) contrast enema B) OGT + duodenoduodenostomy workup C) rectal biopsy D) urgent laparotomy E) observe 6h",
    answer="B", grade="mcq"),
  "med_b2": dict(domain="medicine", par_tokens=1600,
    prompt="Growth velocity for a 3-week preterm 32-weeker is 8.9 g/kg/day. Per NeoBrief protocol band 10-21, what action? Answer with exactly one word: REVIEW or NONE.",
    answer="REVIEW", grade="mcq"),
  # --- trading: reasoning with numeric honesty ---
  "trade_b1": dict(domain="trading", par_tokens=2400,
    prompt="Trade journal composition: 80 wins averaging +0.459R, 36 losses averaging -0.416R, 7 partial wins totaling +0.44R, 28 breakeven scratches at 0.0R; 151 closed entries total. Compute expectancy per trade and profit factor. Show the arithmetic.",
    grade="numeric", answer={"expectancy": 0.147, "profit_factor": 2.48, "tol": 0.03}),
  "trade_b2": dict(domain="trading", par_tokens=2000,
    prompt="A strategy wins 30% with 4R targets and loses -1R. Another wins 75% with -1R stops and +0.2R scratches. Same risk per trade. Which has higher expectancy and why, in 3 sentences.",
    grade="manual_heuristic"),
  # --- longform: adherence + density ---
  "long_b1": dict(domain="longform", par_tokens=2600,
    prompt="In maximum 120 words, explain to a house officer why a 24-week GA baby who died March 4 is a quasi-identifier in a small NICU audit. No bullet lists. No hedging.",
    grade="manual_heuristic"),
}

EFFORT_WRAPPERS = {
  "low": "Answer concisely. Minimal verification. {p}",
  "medium": "{p}",
  "high": "Before answering: restate the constraint, solve, verify by an independent method, list one failure mode you avoided. Then give the final answer. {p}",
}

def call_ollama(model_tag, prompt, reasoning=None, image=None):
    if image:  # multimodal file-reference style for kimi/glm vision
        cmd = ["ollama", "run", model_tag, prompt + f"\n\n[Image attached at {image}]"]
    else:
        cmd = ["ollama", "run", model_tag, prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    out = p.stdout.strip()
    thinking = ""
    m = re.search(r"Thinking\.\.\.(.*?)\.\.\.done thinking\.", out, re.S)
    if m:
        thinking = m.group(1).strip()
        out = re.sub(r"Thinking\.\.\..*?done thinking\.\n*", "", out, flags=re.S).strip()
    return {"text": out, "thinking_chars": len(thinking), "latency_s": round(dt, 1)}

def call_codex(model_flag, prompt):
    t0 = time.time()
    args = ["codex", "exec", "--skip-git-repo-check"] + (["-m", model_flag] if model_flag else []) + [prompt]
    p = subprocess.run(args, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    return {"text": p.stdout.strip(), "thinking_chars": 0, "latency_s": round(dt, 1)}

def sanitize(text):
    text = re.sub(r"\x1b\[[?0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"[\u2800-\u28ff]", "", text)  # braille spinner chars
    return text

def tokens_estimate(text):
    # deterministic proxy when provider token counts unavailable: ~4 chars/token, floored
    return max(1, round(len(text) / 4))



KEY_ALIASES = {"expectancy": ["expectancy", "e =", "e =", "per trade"], "profit_factor": ["profit factor", "pf"]}

def grade_numeric(text, spec):
    tl = text.lower().replace(",", "")
    ok = 0.0
    for k, v in spec.items():
        if k == "tol": continue
        found = False
        for alias in KEY_ALIASES.get(k, [k]):
            for m in re.finditer(rf"{alias}[^\n\d-]{{0,24}}(-?\d+\.?\d*)", tl):
                if abs(float(m.group(1)) - v) <= spec["tol"]:
                    found = True; break
            if found: break
        if found: ok += 0.5
    return min(1.0, ok)

def grade_unit(text):
    # prefer fenced code block (immune to ollama TUI line-wrap corruption): exec whole fence
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if fences:
        env = {}
        try:
            exec(fences[0], {"__builtins__": __builtins__}, env)
            solve = env.get("solve")
            if solve:
                cases = [("eceba", 3), ("aa", 2), ("abaccc", 4), ("", 0), ("aaaa", 4), ("abacccacac", 8), ("aabaccbaaa", 4)]
                got = [solve(s) for s, _ in cases]
                want = [w for _, w in cases]
                return 1.0 if got == want else (0.5 if sum(g == w for g, w in zip(got, want)) >= 5 else 0.0)
        except Exception:
            pass  # fall through to regex path
    m = re.search(r"def solve\s*\(.+?\):[^\n]*\n((?:[ \t]+.*\n?|\n)+)", text)
    if not m: return 0.0
    block = m.group(0)
    env = {}
    try:
        exec(block, {"__builtins__": __builtins__}, env)
        solve = env.get("solve")
        if not solve: return 0.0
        cases = [("eceba", 3), ("aa", 2), ("abaccc", 4), ("", 0), ("aaaa", 4), ("abacccacac", 8), ("aabaccbaaa", 4)]
        got = [solve(s) for s, _ in cases]
        want = [w for _, w in cases]
        return 1.0 if got == want else (0.5 if sum(g == w for g, w in zip(got, want)) >= 3 else 0.0)
    except Exception:
        return 0.0


def grade_dispatch(task, text):
    t = TASKS[task]
    if t["grade"] == "unit": return grade_unit(text)
    if t["grade"] == "mcq": return 1.0 if re.search(rf"\b{t['answer']}\b", text.upper()) else 0.0
    if t["grade"] == "numeric": return grade_numeric(text, t["answer"])
    if t["grade"] in ("manual_heuristic", "checklist"):
        return None  # queued for panel/manual scoring pass
    return None

def run(models=None, suite="quick", reasoning=None):
    if suite == "quick":
        chosen = ["code_b1", "trade_b1", "med_b1", "long_b1"]
    else:
        chosen = list(TASKS)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = {"suite": suite, "reasoning": reasoning or "default", "stamp": stamp, "runs": []}
    for mid in (models or [m["id"] for m in ROSTER["models"]]):
        entry = next(m for m in ROSTER["models"] if m["id"] == mid)
        for tid in chosen:
            t = dict(TASKS[tid])
            eff = reasoning or "medium"
            t["prompt"] = EFFORT_WRAPPERS.get(eff, "{p}").format(p=t["prompt"])
            res = {"task": tid, "domain": t["domain"], "model": mid, "reasoning": eff}
            try:
                prov = entry.get("provider", "")
                if prov.startswith("openai"):
                    r = call_codex(entry["command"].split()[-1] if "-m" in entry["command"] else None, t["prompt"])
                    r["text"] = sanitize(r["text"])
                elif "opencode" in prov:
                    t0 = time.time()
                    p2 = subprocess.run(["opencode", "run", "-m", "opencode/gemini-3.8-flash"],
                                        input=t["prompt"], capture_output=True, text=True, timeout=600)
                    dt = time.time() - t0
                    r = {"text": sanitize(p2.stdout.strip()), "thinking_chars": 0, "latency_s": round(dt, 1)}
                else:
                    tag = entry["command"].split()[2]
                    r = call_ollama(tag, t["prompt"], reasoning)
                    r["text"] = sanitize(r["text"])
                res = {"model": mid, "task": tid, "text": r["text"][:4000], "latency_s": r["latency_s"], "thinking_chars": r["thinking_chars"], "tokens_est": tokens_estimate(r["text"]) + r["thinking_chars"] // 4}
                res["correctness"] = grade_dispatch(tid, r["text"])
                res["token_eff"] = None if res["correctness"] in (0.0, None) else max(0.0, min(1.0, 1 - (res["tokens_est"] / t["par_tokens"] - 1))) if ("tokens_est" in res) else None
                out["runs"].append(res)
                print(f"{mid:22} {tid:9} corr={res['correctness']} lat={r['latency_s']}s tok~{res.get('tokens_est')}")
            except Exception as e:
                out["runs"].append({"model": mid, "task": tid, "error": str(e)[:200]})
                print(f"{mid:22} {tid:9} ERROR {str(e)[:80]}")
    path = f"{RESULTS}/{stamp}_{suite}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("saved:", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="quick"); ap.add_argument("--models"); ap.add_argument("--reasoning")
    args = ap.parse_args()
    suite = args.suite
    run(args.models.split(",") if args.models else None, suite, args.reasoning)