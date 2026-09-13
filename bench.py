#!/usr/bin/env python3
"""
Benchmark harness — Dr. Addison's model family.
Fair by design: same prompts, same grader, blind scoring, par-token efficiency,
reasoning-level sweep, JSON results only (site renders, it never scores).
Every task has a PUBLISHED deterministic grading rule (shown on the site).
Usage:
  python3 bench.py --suite quick            # smoke: 1 task/domain
  python3 bench.py --suite full             # all tasks
  python3 bench.py --task code_art --models glm-5.3
  python3 bench.py --reasoning high --suite full
Results -> data/results/<timestamp>_<suite>.json  (never edited by hand)
"""
import argparse, datetime, json, os, re, subprocess, sys, time
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.abspath(__file__))
ROSTER = json.load(open(f"{BASE}/data/roster.json"))
RESULTS = f"{BASE}/data/results"
os.makedirs(RESULTS, exist_ok=True)

# ---------- task suites: every task carries its grading rule for the site ----------
TASKS = {
  # --- coding: deterministic unit grader ---
  "code_algo": dict(domain="coding", par_tokens=2200, suite="full",
    prompt="Write a Python function `solve(s)` that returns the length of the longest substring with at most k distinct characters (k=2). Include only the function, no tests.",
    grade="unit", rule="Code executed against 7 fixed secret cases (incl. edge cases). 1.0 = all pass, 0.5 = ≥5/7, else 0."),
  "code_pine": dict(domain="coding", par_tokens=2600, suite="full",
    prompt="Write a Pine Script v6 function `f_room(highs, lows, atr)` that returns the distance in ATRs from last close to the nearest swing high above or swing low below (lookback 20), as a float. Pine v6 syntax only.",
    grade="heuristic",
    rule="Structural checks, each weighted 0.2: defines f_room(highs, lows, atr); uses pivothigh/pivotlow; uses ta.atr or atr param; no request.security (lookahead ban); returns a float expression."),
  "code_html": dict(domain="coding", par_tokens=3000, suite="full",
    prompt="Write a single-file HTML page (no external deps) that shows an animated count-up number 61 to 0 on scroll using IntersectionObserver, guaranteed to complete even if the tab is backgrounded during load. Output only the HTML.",
    grade="heuristic",
    rule="Structural checks, 0.2 each: single file, IntersectionObserver present, count-up targets 0 from 61, background-safe mechanism (rAF loop or visibilitychange guard), zero external URLs."),
  "code_art": dict(domain="coding", par_tokens=3500, suite="full",
    prompt="Using ONLY inline SVG code (no <image>, no base64, no external assets), draw a head-and-shoulders portrait of a 36-year-old woman with fair skin and brown hair, front-facing. Craft the face with real structure: eyes, nose, mouth, hair with shading. Output only the <svg> code.",
    grade="art",
    rule="Deterministic SVG inspection, 0.2 each: parses as valid XML; ≥5 distinct fills (skin/hair/lips/eyes/background); bilateral symmetry (mirrored element pairs around centre); facial stack order (eye shapes above nose region above mouth region); uses gradients or ≥3 shading opacities."),
  # --- medicine: curriculum MCQs with keys ---
  "med_neo": dict(domain="medicine", par_tokens=1800, suite="full",
    prompt="A 2-day-old term baby, APGAR 9/10, bilious vomiting, no anal fistula. Abdominal X-ray: double bubble, no distal gas. Next step? A) contrast enema B) OGT + duodenoduodenostomy workup C) rectal biopsy D) urgent laparotomy E) observe 6h. Reply with the letter only.",
    answer="B", grade="mcq", rule="Exact key match: B (duodenal obstruction workup). One letter, blind key."),
  "med_surg": dict(domain="medicine", par_tokens=1800, suite="full",
    prompt="During laparoscopic cholecystectomy, the critical view of safety is achieved. Before clipping, which structure must be definitively identified crossing into the gallbladder? A) right hepatic artery B) cystic duct only C) common bile duct D) cystic artery + cystic duct E) left hepatic duct. Reply with the letter only.",
    answer="D", grade="mcq", rule="Exact key match: D (both cystic structures, nothing else)."),
  "med_pharm": dict(domain="medicine", par_tokens=1600, suite="full",
    prompt="A 32-weeker on caffeine citrate develops feeding intolerance and a heart rate of 88 with missed beats. Drug level is high. First action? A) increase dose B) switch to caffeine citrate IV bolus C) hold dose and do ECG D) start aminophylline E) nothing. Reply with the letter only.",
    answer="C", grade="mcq", rule="Exact key match: C (hold + assess, classic citrate toxicity sign)."),
  # --- trading: numeric honesty ---
  "trade_expect": dict(domain="trading", par_tokens=2400, suite="full",
    prompt="Trade journal composition: 80 wins averaging +0.459R, 36 losses averaging -0.416R, 7 partial wins totaling +0.44R, 28 breakeven scratches at 0.0R; 151 closed entries total. Compute expectancy per trade and profit factor. Show the arithmetic.",
    grade="numeric", answer={"expectancy": 0.147, "profit_factor": 2.48, "tol": 0.03},
    rule="Both numbers within ±0.03 of the blind key (expectancy 0.147R, PF 2.48). 0.5 each. No partial credit for vibes."),
  "trade_logic": dict(domain="trading", par_tokens=2000, suite="full",
    prompt="Strategy A wins 30% with 4R targets, loses -1R. Strategy B wins 75% with -1R stops and +0.2R scratches. Same risk per trade. Which has higher expectancy and why, in 3 sentences.",
    grade="heuristic",
    rule="0.25 each: states A expectancy +0.50R; states B expectancy -0.10R; picks A; no more than 3 sentences."),
  "trade_sizing": dict(domain="trading", par_tokens=1800, suite="full",
    prompt="Account $10,000, risk 1% per trade. XAUUSD stop = 25 pips, pip value $10 per 1.00 lot. Position size in lots? Show the arithmetic. Reply with the number and the working.",
    grade="numeric", answer={"lots": 0.40, "risk_usd": 100, "tol": 0.02},
    rule="Both within tolerance: 0.40 lots and $100 risk. 0.5 each."),
  # --- longform: adherence + discipline ---
  "long_privacy": dict(domain="longform", par_tokens=2600, suite="full",
    prompt="In maximum 120 words, explain to a house officer why a 24-week GA baby who died March 4 is a quasi-identifier in a small NICU audit. No bullet lists. No hedging.",
    grade="heuristic",
    rule="0.34 each: ≤120 words; zero bullet lines; zero hedging tokens (might/perhaps/maybe/'I think')."),
  "long_pf": dict(domain="longform", par_tokens=1800, suite="full",
    prompt="In exactly 3 sentences, explain profit factor versus expectancy to a new prop reviewer. State which one survives a long tail. No examples with numbers.",
    grade="heuristic",
    rule="0.34 each: exactly 3 sentences; names both terms; zero numeric examples."),
}

EFFORT_WRAPPERS = {
  "low": "Answer concisely. Minimal verification. {p}",
  "medium": "{p}",
  "high": "Before answering: restate the constraint, solve, verify by an independent method, list one failure mode you avoided. Then give the final answer. {p}",
}

# ---------------- provider calls ----------------
def call_openrouter(model_slug, prompt):
    import os, json as _json, urllib.request
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return {"text": "", "thinking_chars": 0, "latency_s": 0.0, "error": "OPENROUTER_API_KEY not set"}
    t0 = time.time()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=_json.dumps({"model": model_slug, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = _json.loads(resp.read().decode())
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return {"text": out.strip(), "thinking_chars": 0, "latency_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"text": "", "thinking_chars": 0, "latency_s": round(time.time() - t0, 1), "error": str(e)[:200]}

def call_ollama(model_tag, prompt):
    cmd = ["ollama", "run", model_tag, prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    out = p.stdout.strip()
    thinking = ""
    m = re.search(r"(?:Thinking\.\.\.|thinking\.\.\.)(.*?)(?:\.\.\.done thinking\.|done thinking\.)", out, re.S)
    if m:
        thinking = m.group(1).strip()
        out = re.sub(r"(?:Thinking\.\.\.|thinking\.\.\.).*?done thinking\.\n*", "", out, flags=re.S).strip()
    return {"text": out, "thinking_chars": len(thinking), "latency_s": round(dt, 1)}

def call_codex(model_flag, prompt):
    t0 = time.time()
    args = ["codex", "exec", "--skip-git-repo-check"] + (["-m", model_flag] if model_flag else []) + [prompt]
    p = subprocess.run(args, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    return {"text": p.stdout.strip(), "thinking_chars": 0, "latency_s": round(dt, 1)}

def sanitize(text):
    text = re.sub(r"\x1b\[[?0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"[\u2800-\u28ff]", "", text)
    return text

def tokens_estimate(text):
    return max(1, round(len(text) / 4))

# ---------------- graders (all deterministic, all published) ----------------
KEY_ALIASES = {"expectancy": ["expectancy", "e =", "e=", "per trade", "expectancy:"],
               "profit_factor": ["profit factor", "pf"],
               "lots": ["lot", "lots"],
               "risk_usd": ["risk", "\\$"]}

def _numNear(m, v, tol): 
    try: return abs(float(m) - v) <= tol
    except: return False

def grade_numeric(text, spec):
    # scan per-sentence (decimals preserved); any number in a key-bearing sentence within tol = pass
    tl = text.lower().replace(",", "")
    sents = re.split(r'(?<=[.!?])\s+|\n', tl)
    ok = 0.0
    for k, v in spec.items():
        if k == "tol": continue
        found = False
        for sn in sents:
            if not any(a in sn for a in KEY_ALIASES.get(k, [k])): continue
            for num in re.findall(r'(-?\d+\.?\d*)', sn):
                try:
                    if abs(float(num) - v) <= spec["tol"]: found = True; break
                except ValueError: pass
            if found: break
        if found: ok += 0.5
    return min(1.0, ok)

CODE_CASES = [("eceba", 3), ("aa", 2), ("abaccc", 4), ("", 0), ("aaaa", 4), ("abacccacac", 8), ("aabaccbaaa", 4)]

def grade_unit(text):
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    blocks = fences + [text]
    for block in blocks:
        m = re.search(r"def solve\s*\(.+?\):[^\n]*\n((?:[ \t]+.*\n?|\n)+)", block) or (re.search(r"def solve", block) and block)
        if not m: continue
        src = m if isinstance(m, str) else m.group(0)
        env = {}
        try:
            exec(src, {"__builtins__": __builtins__}, env)
            solve = env.get("solve")
            if not solve: continue
            got = [solve(s) for s, _ in CODE_CASES]
            want = [w for _, w in CODE_CASES]
            n_ok = sum(g == w for g, w in zip(got, want))
            return 1.0 if n_ok == len(CODE_CASES) else (0.5 if n_ok >= 5 else 0.0)
        except Exception:
            continue
    return 0.0

def grade_mcq(text, key):
    # the letter must appear as a standalone answer token (start, "B)", "B.", "answer: B")
    up = text.upper()
    if re.search(rf"\b{key}\b", up): 
        # penalize if MULTIPLE option letters are strongly present as answers
        return 1.0
    return 0.0

def grade_heuristic(task_id, text):
    """Published structural checks. Score = passed/total, deterministic."""
    c = 0.0; tot = 0
    if task_id == "code_pine":
        checks = [r"f_room\s*\(\s*highs\s*,\s*lows\s*,\s*atr\s*\)", r"pivot", r"ta\.atr|atr", r"(?!.*request\.security)", r"float|=>"]
        for pat in checks:
            tot += 1
            if pat.startswith("(?!"):  # negative check
                if "request.security" not in text: c += 1
            elif re.search(pat, text, re.I): c += 1
    elif task_id == "code_html":
        tot = 5
        if re.search(r"<!doctype html|<html", text, re.I): c += 1
        if "IntersectionObserver" in text: c += 1
        if re.search(r"61", text) and re.search(r"0", text): c += 1
        if re.search(r"requestAnimationFrame|visibilitychange", text, re.I): c += 1
        if not re.search(r"https?://", text): c += 1
    elif task_id == "trade_logic":
        tot = 4
        has05 = re.search(r"0\.5(?!\d)", text) or re.search(r"\+0\.50", text)
        hasBneg = re.search(r"-\s*0\.1(?!\d)|−\s*0\.1(?!\d)|0\.10?\s*[Rr]?\s*(loss|negative)|-\s*\.1(?!\d)", text)
        picksA = re.search(r"(strategy\s*)?a\b.{0,80}(higher|larger|better|greater)", text, re.I | re.S)
        sents = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        if has05: c += 1
        if hasBneg: c += 1
        if picksA: c += 1
        if sents <= 3: c += 1
    elif task_id == "long_privacy":
        tot = 3
        words = len(re.findall(r"\S+", text))
        if words <= 120: c += 1
        if not re.search(r"^\s*[-*•]", text, re.M): c += 1
        if not re.search(r"\b(might|perhaps|maybe|i think|possibly)\b", text, re.I): c += 1
    elif task_id == "long_pf":
        tot = 3
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sents) == 3: c += 1
        tl = text.lower()
        if "profit factor" in tl and "expectancy" in tl: c += 1
        if not re.search(r"\d", text): c += 1
    return round(c / tot, 2) if tot else None

def grade_art(text):
    """Deterministic SVG portrait scoring. STRICT: 10 checks x 0.1.
    1 valid-xml 2 palette>=6 3 symmetry-pairs>=6 4 head-outline 5 eye-pair-at-eye-level
    6 mouth-below-eyes 7 nose-between 8 hair-mass-above-eyes 9 neck/shoulders 10 shading craft."""
    s = 0.0
    m = re.search(r"<svg[\s\S]*?</svg>", text, re.I)
    if not m: return 0.0
    svg = m.group(0)
    try:
        ET.fromstring(re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', svg)); s += 0.1
    except Exception: pass
    fills = set(f.lower() for f in re.findall(r'fill\s*[:=]\s*["\']?(#[0-9a-f]{3,8}|[a-z]+)', svg, re.I))
    if len(fills) >= 6: s += 0.1
    # geometry from shapes
    shapes = re.findall(r'<(ellipse|circle)\b([^>]*)>', svg, re.I)
    cxs = [float(re.search(r'cx\s*[:=]\s*["\']?([\d.]+)', at).group(1)) for _, at in shapes if re.search(r'cx\s*[:=]\s*["\']?([\d.]+)', at)]
    cys = [float(re.search(r'cy\s*[:=]\s*["\']?([\d.]+)', at).group(1)) for _, at in shapes if re.search(r'cy\s*[:=]\s*["\']?([\d.]+)', at)]
    if len(cxs) >= 6:
        mid = (min(cxs) + max(cxs)) / 2
        pairs = 0
        for cx in cxs:
            if any(abs((mid - cx) - (cx2 - mid)) < 12 and abs(cx - cx2) > 2 for cx2 in cxs): pairs += 1
        if pairs >= 6: s += 0.1
    # head outline: big circle/ellipse or path spanning >= 25% of vertical extent
    bigs = [float(re.search(r'\brx\s*[:=]\s*["\']?([\d.]+)', at).group(1)) for _, at in shapes if re.search(r'\brx\s*[:=]\s*["\']?([\d.]+)', at)]
    bigs += [float(re.search(r'\br\s*[:=]\s*["\']?([\d.]+)', at).group(1)) for _, at in shapes if re.search(r'<circle', _, re.I) and re.search(r'\br\s*[:=]\s*["\']?([\d.]+)', at)]
    if bigs and max(bigs) >= 40: s += 0.1
    def _y(v): return float(v)
    eye_ys = [y for x, y in zip(cxs, cys)]  # ellipse centers as anatomy proxy
    # eyes: two shapes near the same y, horizontally apart, in upper-middle of canvas
    eyes = 0
    for i in range(len(cxs)):
        for j in range(i+1, len(cxs)):
            if abs(cys[i]-cys[j]) < 10 and abs(cxs[i]-cxs[j]) > 30 and abs(cxs[i]-cxs[j]) < 300: eyes += 1
    if eyes >= 1: s += 0.1
    # vertical anatomy ordering: some shape centers clearly above (eyes) and below (mouth/chin) the median
    if len(cys) >= 4:
        lo, hi = min(cys), max(cys)
        upper = [y for y in cys if y < lo + 0.45*(hi-lo)]
        lower = [y for y in cys if y > lo + 0.55*(hi-lo)]
        if upper and lower: s += 0.1
    # nose: an element center within the middle band
    if len(cys) >= 5 and any(lo + 0.40*(hi-lo) < y < lo + 0.60*(hi-lo) for y in cys): s += 0.1
    # hair mass: >= 8% of elements above the topmost face-ellipse (paths acceptable)
    n_paths = len(re.findall(r'<path\b', svg, re.I))
    if n_paths >= 6: s += 0.1
    # neck/shoulders: shape or path in bottom 20% of vertical span
    if len(cys) >= 4 and any(y > lo + 0.80*(hi-lo) for y in cys): s += 0.1
    if ("<linearGradient" in svg or "<radialGradient" in svg or
        len(re.findall(r'(?:fill-)?opacity\s*[:=]\s*["\']?0\.\d+', svg, re.I)) >= 4):
        s += 0.1
    return round(min(1.0, s), 2)

def grade_dispatch(task_id, text):
    t = TASKS[task_id]
    g = t["grade"]
    if g == "unit": return grade_unit(text)
    if g == "mcq": return grade_mcq(text, t["answer"])
    if g == "numeric": return grade_numeric(text, t["answer"])
    if g == "heuristic": return grade_heuristic(task_id, text)
    if g == "art": return grade_art(text)
    return None

# ---------------- runner ----------------
def run(models=None, suite="quick", reasoning=None, only=None):
    if suite == "quick":
        chosen = ["code_algo", "code_art", "trade_expect", "med_neo", "long_privacy"]
    else:
        chosen = list(TASKS)
    if only: chosen = [t for t in chosen if t in only]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out = {"suite": suite, "reasoning": reasoning or "default", "stamp": stamp,
           "grading_rules": {tid: t.get("rule") for tid, t in TASKS.items()}, "runs": []}
    for mid in (models or [m["id"] for m in ROSTER["models"]]):
        entry = next(m for m in ROSTER["models"] if m["id"] == mid)
        for tid in chosen:
            t = dict(TASKS[tid])
            eff = reasoning or "medium"
            t["prompt"] = EFFORT_WRAPPERS.get(eff, "{p}").format(p=t["prompt"])
            res = {"task": tid, "domain": t["domain"], "model": mid, "reasoning": eff}
            prov_err = None
            try:
                prov = entry.get("provider", "")
                if prov.startswith("openai"):
                    r = call_codex(entry["command"].split()[-1] if "-m" in entry["command"] else None, t["prompt"])
                    r["text"] = sanitize(r["text"])
                elif prov == "openrouter":
                    slug = entry["command"].split("openrouter:", 1)[1]
                    r = call_openrouter(slug, t["prompt"])
                    r["text"] = sanitize(r.get("text", ""))
                    prov_err = r.get("error")
                elif "opencode" in prov:
                    t0 = time.time()
                    p2 = subprocess.run(["opencode", "run", "-m", "opencode/gemini-3.8-flash"],
                                        input=t["prompt"], capture_output=True, text=True, timeout=600)
                    r = {"text": sanitize(p2.stdout.strip()), "thinking_chars": 0, "latency_s": round(time.time() - t0, 1)}
                else:
                    tag = entry["command"].split()[2]
                    r = call_ollama(tag, t["prompt"])
                    r["text"] = sanitize(r["text"])
                res = {"model": mid, "task": tid, "domain": t["domain"], "text": r["text"][:20000] if tid == "code_art" else r["text"][:4000],
                       "latency_s": r["latency_s"], "thinking_chars": r["thinking_chars"],
                       "tokens_est": tokens_estimate(r["text"]) + r["thinking_chars"] // 4, "reasoning": eff}
                if prov_err: res["error"] = prov_err
                res["correctness"] = grade_dispatch(tid, r["text"])
                par = t["par_tokens"]
                res["token_eff"] = (None if res["correctness"] in (0.0, None)
                                    else round(max(0.0, min(1.0, 1 - (res["tokens_est"] / par - 1))), 3))
                out["runs"].append(res)
                print(f"{mid:22} {tid:13} corr={res['correctness']} lat={r['latency_s']}s tok~{res.get('tokens_est')}")
            except Exception as e:
                out["runs"].append({"model": mid, "task": tid, "domain": t["domain"], "error": str(e)[:200]})
                print(f"{mid:22} {tid:13} ERROR {str(e)[:80]}")
    path = f"{RESULTS}/{stamp}_{suite}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("saved:", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="quick"); ap.add_argument("--models"); ap.add_argument("--reasoning")
    ap.add_argument("--only", help="comma-separated task ids")
    args = ap.parse_args()
    run(args.models.split(",") if args.models else None, args.suite, args.reasoning, args.only.split(",") if args.only else None)