# -*- coding: utf-8 -*-

import os, re, json, csv, time, argparse, concurrent.futures, threading, base64, mimetypes, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import requests


ERROR_THRESHOLD = 100
PAUSE_DURATION_SECONDS = 20 * 60
IO_SEMA = threading.Semaphore(4)

class ApiProvider:
    def __init__(self, name, base_url, api_key, model, max_workers=12, timeout=(10,180)):
        self.name = name
        self.url = f"{base_url.strip('/')}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self.lock = threading.Lock()
        self.consecutive_errors = 0
        self.paused_until = 0

    def headers(self):
        return {"Content-Type":"application/json","Authorization":f"Bearer {self.api_key}"}

    def ping(self):
        try:
            r = requests.post(self.url, headers=self.headers(),
                              json={"model": self.model, "messages":[{"role":"user","content":"ping"}], "max_tokens": 8},
                              timeout=(5,10))
            r.raise_for_status()
            tqdm.write(f"[PING OK] {self.name}")
            return True
        except Exception as e:
            tqdm.write(f"[PING FAIL] {self.name} -> {e}")
            return False

    def is_paused(self):
        if self.paused_until and time.time() < self.paused_until:
            return True
        if self.paused_until and time.time() >= self.paused_until:
            with self.lock:
                self.paused_until = 0
            tqdm.write(f"[INFO] Provider '{self.name}' resumed.")
        return False

    def record_success(self):
        with self.lock:
            self.consecutive_errors = 0

    def record_failure(self):
        with self.lock:
            self.consecutive_errors += 1
            if self.consecutive_errors >= ERROR_THRESHOLD:
                self.paused_until = time.time() + PAUSE_DURATION_SECONDS
                tqdm.write(f"\n[CRITICAL] '{self.name}' paused {PAUSE_DURATION_SECONDS/60:.0f} min (too many errors)")
                self.consecutive_errors = 0


SKINCON_SET = [
    "Abscess","Acuminate","Atrophy","Black","Blue","Brown(Hyperpigmentation)","Bulla","Burrow","Comedo","Crust","Cyst",
    "Dome-shaped","Erosion","Erythema","Excoriation","Exophytic/Fungating","Exudate","Fissure","Flat topped","Friable",
    "Gray","Induration","Lichenification","Macule","Nodule","Papule","Patch","Pedunculated","Pigmented","Plaque",
    "Poikiloderma","Purple","Purpura/Petechiae","Pustule","Salmon","Scale","Scar","Sclerosis","Telangiectasia",
    "Translucent","Ulcer","Umbilicated","Vesicle","Warty/Papillomatous","Wheal","White(Hypopigmentation)","Xerosis","Yellow"
]

D7_KEYS = [
    "pigment_network","blue_whitish_veil","vascular_structures",
    "pigmentation","streaks","dots_and_globules","regression_structures"
]
D7_ENUM = {
    "pigment_network": ["absent","typical","atypical"],
    "blue_whitish_veil": ["absent","present"],
    "vascular_structures": ["absent","arborizing","comma","hairpin","within regression","wreath","dotted","linear irregular"],
    "pigmentation": ["absent","diffuse regular","localized regular","diffuse irregular","localized irregular"],
    "streaks": ["absent","regular","irregular"],
    "dots_and_globules": ["absent","regular","irregular"],
    "regression_structures": ["absent","blue areas","white areas","combinations"]
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def parse_first_json_in_morph_block(s: str) -> Tuple[str, dict]:
    s = s or ""
    tag = re.search(r"<morph>(.*?)</morph>", s, flags=re.S|re.I)
    scope = tag.group(1) if tag else s
    j = re.search(r"\{.*?\}", scope, flags=re.S)
    if not j: return ("", None)
    raw = j.group(0)
    try:
        obj = json.loads(raw)
        return (raw, obj)
    except Exception:
        return (raw, None)

def detect_schema(obj: dict) -> str:
    if isinstance(obj, dict):
        if "morphological_features_Derm7pt" in obj: return "Derm7pt"
        if "morphological_features_skincon" in obj: return "SkinCon"
    return "Unknown"


def skincon_syntax_metrics(gt_obj: dict, ca_obj: dict) -> dict:
    out = {
        "schema": "SkinCon",
        "json_valid_reference": isinstance(gt_obj, dict),
        "json_valid_candidate": isinstance(ca_obj, dict),
        "syntactic_precision": 0.0, "syntactic_recall": 0.0, "syntactic_f1": 0.0,
        "candidate_coverage": 0.0,
        "invalid_items_candidate": []
    }
    if not isinstance(gt_obj, dict) or not isinstance(ca_obj, dict):
        return out
    g = gt_obj.get("morphological_features_skincon", [])
    c = ca_obj.get("morphological_features_skincon", [])
    if not isinstance(g, list) or not isinstance(c, list):
        return out
    gset = set(_norm(x) for x in g)
    cset = set(_norm(x) for x in c)
    allow = set(_norm(x) for x in SKINCON_SET)
    invalid = [x for x in c if _norm(x) not in allow]
    tp = len(gset & cset)
    prec = tp / max(1, len(cset))
    rec  = tp / max(1, len(gset))
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    out.update({
        "syntactic_precision": round(prec,4),
        "syntactic_recall": round(rec,4),
        "syntactic_f1": round(f1,4),
        "candidate_coverage": round(len([x for x in c if _norm(x) in allow])/max(1,len(c)),4),
        "invalid_items_candidate": invalid
    })
    return out

def derm7pt_syntax_metrics(gt_obj: dict, ca_obj: dict) -> dict:
    out = {
        "schema": "Derm7pt",
        "json_valid_reference": isinstance(gt_obj, dict),
        "json_valid_candidate": isinstance(ca_obj, dict),
        "has_all_keys_reference": False,
        "has_all_keys_candidate": False,
        "syntactic_accuracy": 0.0,
        "field_coverage_candidate": 0.0,
        "invalid_fields_candidate": []
    }
    if not isinstance(gt_obj, dict) or not isinstance(ca_obj, dict):
        return out
    gt_inner = gt_obj.get("morphological_features_Derm7pt", {})
    ca_inner = ca_obj.get("morphological_features_Derm7pt", {})
    if not isinstance(gt_inner, dict) or not isinstance(ca_inner, dict):
        return out

    out["has_all_keys_reference"] = all(k in gt_inner for k in D7_KEYS)
    out["has_all_keys_candidate"] = all(k in ca_inner for k in D7_KEYS)

    total, correct = 0, 0
    valid_count, valid_total = 0, len(D7_KEYS)
    invalids = []
    for k in D7_KEYS:
        gv = _norm(gt_inner.get(k))
        cv = _norm(ca_inner.get(k))
        total += 1
        if cv in [_norm(x) for x in D7_ENUM[k]]:
            valid_count += 1
        else:
            invalids.append(k)
        if gv == cv and cv in [_norm(x) for x in D7_ENUM[k]]:
            correct += 1

    out["syntactic_accuracy"] = round(correct / max(1,total), 4)
    out["field_coverage_candidate"] = round(valid_count / max(1,valid_total), 4)
    out["invalid_fields_candidate"] = invalids
    return out

def syntax_metrics_dispatch(gt_obj: dict, ca_obj: dict) -> dict:
    sch = detect_schema(gt_obj)
    if sch == "SkinCon":
        m = skincon_syntax_metrics(gt_obj, ca_obj)
        m["syntactic_auto_0_1"] = m.get("syntactic_f1", 0.0)
        return m
    if sch == "Derm7pt":
        m = derm7pt_syntax_metrics(gt_obj, ca_obj)
        m["syntactic_auto_0_1"] = m.get("syntactic_accuracy", 0.0)
        return m

    return {"schema":"Unknown","json_valid_reference":bool(gt_obj),"json_valid_candidate":bool(ca_obj),
            "syntactic_precision":0.0,"syntactic_recall":0.0,"syntactic_f1":0.0,
            "syntactic_accuracy":0.0,"syntactic_auto_0_1":0.0}


SYSTEM_PROMPT_TASK32 = (
    "You are a strict dermatology evaluator for Task 3.2 (reasoning + morph JSON + final diagnosis). "
    "You DO NOT see the image. Focus on CONTENT, not formatting. "
    "Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. "
    "Do NOT penalize missing tags, extra whitespace, or ordering differences. "
    "If a JSON object appears anywhere, treat the FIRST JSON object as the morph content. "
    "If no JSON is present, infer the morph feature set from the surrounding text. "
    "SCHEMA SELECTION RULE: Detect the schema used by REFERENCE. Compare and output using the SAME schema."
)

USER_TEMPLATE_TASK32 = """You will be given REFERENCE and CANDIDATE texts containing three conceptual parts: <reasoning>, <morph> JSON, and <final_diagnosis>.
Be format-agnostic; extract content even when tags are missing or order differs.

Allowed schemas:
- Derm7pt (object with EXACT keys):
  pigment_network: absent|typical|atypical
  blue_whitish_veil: absent|present
  vascular_structures: absent|arborizing|comma|hairpin|within regression|wreath|dotted|linear irregular
  pigmentation: absent|diffuse regular|localized regular|diffuse irregular|localized irregular
  streaks: absent|regular|irregular
  dots_and_globules: absent|regular|irregular
  regression_structures: absent|blue areas|white areas|combinations

- SkinCon (array of strings only):
  {{\"morphological_features_skincon\": [ ... ]}}  where each item is from this CLOSED set (case-sensitive):
  Abscess, Acuminate, Atrophy, Black, Blue, Brown(Hyperpigmentation), Bulla, Burrow, Comedo, Crust, Cyst, Dome-shaped, Erosion, Erythema, Excoriation, Exophytic/Fungating, Exudate, Fissure, Flat topped, Friable, Gray, Induration, Lichenification, Macule, Nodule, Papule, Patch, Pedunculated, Pigmented, Plaque, Poikiloderma, Purple, Purpura/Petechiae, Pustule, Salmon, Scale, Scar, Sclerosis, Telangiectasia, Translucent, Ulcer, Umbilicated, Vesicle, Warty/Papillomatous, Wheal, White(Hypopigmentation), Xerosis, Yellow.

SCHEMA SELECTION:
- Detect the schema used by REFERENCE (Derm7pt vs SkinCon). Use that schema for extraction/normalization and comparison. Do NOT switch schemas.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Tasks:

A) EXTRACTION:
  Extract for both REFERENCE and CANDIDATE:
   - reasoning: concise rationale text (from <reasoning> if present; else best-effort).
   - morph: a normalized JSON in the SAME schema as the REFERENCE:
       • If Derm7pt: output an object with ALL seven keys, each with ONE allowed value (lowercase).
       • If SkinCon: output exactly {{\"morphological_features_skincon\": [ ... ]}} where items are ONLY from the closed set; normalize synonyms to the closest allowed token; sort alphabetically.
   - final_dx: single most likely diagnosis (free-text term).

B) REASONING ALIGNMENT (lenient v2):
  - Decompose GT reasoning into ≤25 atomic claims (morphology, logic, differentials).
  - For each claim, label CANDIDATE reasoning as: Supported | PartiallySupported | Vague | Missing | Contradicted.
  - Count ExtraIncorrect: only specific, materially false statements (exclude harmless generalities).
  - Scoring:
    Let T = max(1, total_ref_claims).
    recall_like = (1.0*Supported + 0.6*PartiallySupported + 0.3*Vague) / T
    contrad_pen = 0.7*Contradicted / T
    missing_pen = 0.4*Missing / T
    extra_pen   = 0.5*ExtraIncorrect / T
    reasoning_score [0-100] = round(100 * max(0.0, recall_like - 0.5*contrad_pen - 0.2*missing_pen - 0.2*extra_pen), 1)

C) MORPH SEMANTICS:
  - Compare CANDIDATE vs GT morph (after normalization) under the detected schema.
  - Count supported/missing/contradicted/extra (for Derm7pt: per-field; for SkinCon: set comparison).
  - score_semantic in [0,1] reflects degree of agreement. Provide brief notes.

D) DIAGNOSIS SIMILARITY (graded, not binary):
  - Relation between cand_final_dx and gt_final_dx:
    Exact | Synonym | Parent | Child | Sibling/CloseDifferential | SameSuperfamily | UnrelatedPlausible | WrongSystem | Nonsense/NoAnswer
  - Map to similarity s in [0,1]:
    Exact/Synonym=1.0; Parent/Child=0.85; Sibling/CloseDifferential=0.7; SameSuperfamily=0.5; UnrelatedPlausible=0.3; WrongSystem=0.1; Nonsense/NoAnswer=0.0.
  - diagnosis_score [0-100] = round(100 * s, 1).

E) CROSS-CONSISTENCY:
  - Does the CANDIDATE reasoning contradict the CANDIDATE morph JSON? penalty in [0,1] (0=no issue, 1=severe).

STRICT JSON ONLY. Schema:
{{
  "extraction": {{
    "gt": {{"reasoning": "...", "morph": {{}}, "final_dx": "..."}},
    "cand": {{"reasoning": "...", "morph": {{}}, "final_dx": "..."}}
  }},
  "reasoning": {{
    "claims": [{{"text":"...","label":"Supported|PartiallySupported|Vague|Missing|Contradicted"}}],
    "counts": {{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},
    "score": 0.0,
    "notes": "≤60 words"
  }},
  "morph_semantic": {{
    "schema": "SkinCon|Derm7pt",
    "supported": 0, "missing": 0, "contradicted": 0, "extra": 0,
    "score_semantic": 0.0,
    "notes": "≤60 words"
  }},
  "diagnosis": {{
    "gt_dx": "...", "cand_dx": "...",
    "relation": "Exact|Synonym|Parent|Child|Sibling/CloseDifferential|SameSuperfamily|UnrelatedPlausible|WrongSystem|Nonsense/NoAnswer",
    "similarity": 0.0, "score": 0.0,
    "notes": "≤40 words"
  }},
  "cross_consistency": {{"penalty": 0.0, "notes": "≤40 words"}},
  "short_feedback": "≤50 words"
}}
"""


SYSTEM_PROMPT_IMAGE = (
    "You are a dermatologist checking IMAGE CONSISTENCY only. "
    "You will be given an image and two texts: REFERENCE and CANDIDATE. "
    "Ignore wording quality; check whether CANDIDATE contradicts the IMAGE on major morphology axes. "
    "Return STRICT JSON only."
)
USER_TEMPLATE_IMAGE = """You will receive an image and two texts.

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Task: judge ONLY image consistency for the CANDIDATE (the REFERENCE is for context/audit).
Output STRICT JSON:
{{
  "cand_image_penalty": 0.0,
  "ref_image_penalty": 0.0,
  "notes": "≤60 words describing the key contradictions (if any)"
}}
"""


# ------------------- I/O -------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_any(path: Path) -> Dict[str, Dict[str,str]]:
    txt = path.read_text(encoding="utf-8").strip()
    out = {}
    def from_obj(obj):
        _id = obj.get("id")
        conv = obj.get("conversations", [])
        prompt, ans = "", ""
        for t in conv:
            if t.get("from") == "human" and not prompt:
                prompt = t.get("value","")
            if t.get("from") == "gpt":
                ans = t.get("value", ans)
        img = obj.get("image")
        if _id is not None:
            out[_id] = {"prompt":prompt, "answer":ans, "image":img}
    if path.suffix.lower()==".jsonl":
        for line in txt.splitlines():
            if line.strip(): from_obj(json.loads(line))
    else:
        data = json.loads(txt)
        if isinstance(data, list):
            for obj in data: from_obj(obj)
        elif isinstance(data, dict):
            if "conversations" in data: from_obj(data)
            else:
                for _, obj in data.items(): from_obj(obj)
    return out

def write_jsonl(objs: List[dict], path: Path):
    path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in objs), encoding="utf-8")

def write_csv(rows: List[dict], path: Path):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ------------------- API Call -------------------
def _extract_json(txt: str) -> dict:
    cleaned = txt.strip().replace("```json","").replace("```","").strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m: raise ValueError("No JSON object found")
    return json.loads(m.group(0))

def call_chat(provider: ApiProvider, messages: list, max_tokens=10000, temperature=0.0, json_mode=True, retry=3):
    payload = {"model": provider.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for a in range(retry):
        try:
            r = requests.post(provider.url, headers=provider.headers(), json=payload, timeout=provider.timeout)
            r.raise_for_status()
            data = r.json()
            txt = data.get("choices",[{}])[0].get("message",{}).get("content","").strip()
            if not txt: raise ValueError("Empty response")
            obj = _extract_json(txt)
            provider.record_success()
            return obj
        except Exception as e:
            provider.record_failure()
            tqdm.write(f"[RETRY {a+1}/{retry}] {provider.name} -> {e}")
            time.sleep(1.0*(a+1))
    raise RuntimeError("API call failed after retries")


def judge_task32(provider: ApiProvider, task_prompt: str, reference: str, candidate: str) -> dict:
    user = USER_TEMPLATE_TASK32.format(task_prompt=task_prompt.strip(),
                                       reference=reference.strip(),
                                       candidate=candidate.strip())
    return call_chat(
        provider,
        [{"role":"system","content":SYSTEM_PROMPT_TASK32},
         {"role":"user","content":user}],
        temperature=0.0, max_tokens=10000, json_mode=True
    )

def b64_image_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime: mime = "application/octet-stream"
    with IO_SEMA:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def judge_image(provider: ApiProvider, reference: str, candidate: str, image_path: Path) -> dict:
    user_text = USER_TEMPLATE_IMAGE.format(reference=reference.strip(), candidate=candidate.strip())
    url = b64_image_url(image_path)
    messages = [{"role":"system","content":SYSTEM_PROMPT_IMAGE},
                {"role":"user","content":[{"type":"text","text":user_text},
                                          {"type":"image_url","image_url":{"url":url}}]}]
    return call_chat(provider, messages, temperature=0.0, max_tokens=10000, json_mode=False)


def combine_scores(reasoning_0_100: float,
                   dx_0_100: float,
                   morph_sem_0_1: float,
                   morph_syntax_auto_0_1: float,
                   cross_penalty_0_1: float,
                   weights: dict) -> Tuple[float, float, float]:
    wr = weights.get("w_reasoning", 0.5)
    wd = weights.get("w_diagnosis", 0.5)
    w_sem = weights.get("w_morph_semantic", 0.8)
    wt = weights.get("w_text", 0.5)
    wm = weights.get("w_morph", 0.5)
    wc = weights.get("w_cross", 0.0)

    text_block = wr*reasoning_0_100 + wd*dx_0_100
    morph_block = 100.0 * max(0.0, min(1.0, w_sem*morph_sem_0_1 + (1.0-w_sem)*morph_syntax_auto_0_1))
    final_pre = wt*text_block + wm*morph_block
    final = final_pre * (1.0 - max(0.0, min(1.0, wc*cross_penalty_0_1)))
    return (round(text_block,2), round(morph_block,2), round(final,2))

def agg_list_votes(vs: List[dict]) -> dict:
    n = max(1, len(vs))
    reason = sum(float(v.get("reasoning",{}).get("score",0.0)) for v in vs)/n
    dx_score = sum(float(v.get("diagnosis",{}).get("score",0.0)) for v in vs)/n
    dx_sim   = sum(float(v.get("diagnosis",{}).get("similarity",0.0)) for v in vs)/n
    morph_sem = sum(float(v.get("morph_semantic",{}).get("score_semantic",0.0)) for v in vs)/n
    cross_pen = sum(float(v.get("cross_consistency",{}).get("penalty",0.0)) for v in vs)/n
    keys = ["supported","partial","contradicted","missing","vague","extra_incorrect","total_ref_claims"]
    counts = {k: round(sum(float(v.get("reasoning",{}).get("counts",{}).get(k,0.0)) for v in vs)/n) for k in keys}
    return {
        "reasoning_0_100": round(reason,2),
        "dx_0_100": round(dx_score,2),
        "dx_similarity": round(dx_sim,4),
        "morph_sem_0_1": round(morph_sem,4),
        "cross_penalty_0_1": round(cross_pen,4),
        "counts": counts,
        "short_feedback": " | ".join(v.get("short_feedback","") for v in vs if v.get("short_feedback"))[:240]
    }


def run_eval(gt_path: Path, pred_path: Path, provider: ApiProvider,
             out_dir: Path, model_alias: str, n_voters: int, max_workers: int,
             with_image_judge: bool, image_root: Path, image_weight: float,
             overwrite: bool, weights: dict):
    ensure_dir(out_dir)
    items_dir = out_dir / "items"; ensure_dir(items_dir)

    gt_map = load_any(gt_path)
    pred_map = load_any(pred_path)
    ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    tqdm.write(f"[{model_alias}] overlap ids: {len(ids)}")

    def work(_id: str) -> dict:
        cache = items_dir / f"{_id}.json"
        if cache.exists() and not overwrite:
            try: return json.loads(cache.read_text(encoding="utf-8"))
            except Exception: pass

        task_prompt = gt_map[_id]["prompt"]
        ref_full    = gt_map[_id]["answer"]
        cand_full   = pred_map[_id]["answer"]


        ref_raw, ref_obj = parse_first_json_in_morph_block(ref_full)
        cand_raw, cand_obj = parse_first_json_in_morph_block(cand_full)
        syntax = syntax_metrics_dispatch(ref_obj, cand_obj)
        morph_syntax_auto = float(syntax.get("syntactic_auto_0_1", 0.0))


        votes=[]
        for _ in range(max(1, n_voters)):
            res = judge_task32(provider, task_prompt, ref_full, cand_full)
            votes.append(res); time.sleep(0.05)
        agg = agg_list_votes(votes)


        text_block, morph_block, final_overall = combine_scores(
            reasoning_0_100 = agg["reasoning_0_100"],
            dx_0_100 = agg["dx_0_100"],
            morph_sem_0_1 = agg["morph_sem_0_1"],
            morph_syntax_auto_0_1 = morph_syntax_auto,
            cross_penalty_0_1 = agg["cross_penalty_0_1"],
            weights = weights
        )


        img_audit=None; image_path=None
        if with_image_judge:
            rel = gt_map[_id].get("image")
            if rel:
                p = Path(rel); image_path = p if p.is_absolute() else (image_root/rel)
                if image_path.exists():
                    iv=[]
                    for _ in range(max(1,n_voters)):
                        iv.append(judge_image(provider, ref_full, cand_full, image_path)); time.sleep(0.05)
                    pen = sum(float(v.get("cand_image_penalty",0.0)) for v in iv)/max(1,len(iv))
                    img_audit = {"cand_image_penalty": round(pen,3)}
                    final_overall = round(final_overall * (1.0 - image_weight * pen), 2)
                else:
                    tqdm.write(f"[WARN] image not found: id={_id} -> {image_path}")

        rec = {
            "id": _id,
            "task_prompt": task_prompt,
            "reference": ref_full,
            "candidate": cand_full,


            "morph_syntax": syntax,
            "morph_syntax_auto_0_1": morph_syntax_auto,


            "votes": votes,
            "agg": agg,                               # reasoning/dx/morph_sem/cross_penalty
            "text_block_0_100": text_block,
            "morph_block_0_100": morph_block,
            "final_overall": final_overall,


            "image_audit": img_audit,
            "image_path": str(image_path) if image_path else None
        }
        cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec


    rows=[]; per=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(work, _id): _id for _id in ids}
        pbar = tqdm(total=len(ids), desc=f"[{model_alias}] judging", dynamic_ncols=True)
        processed = 0
        for fut in concurrent.futures.as_completed(futs):
            _id = futs[fut]
            try:
                rec = fut.result()
                per.append(rec)
                rows.append({
                    "id": _id,
                    "reasoning": rec["agg"]["reasoning_0_100"],
                    "dx": rec["agg"]["dx_0_100"],
                    "dx_similarity": rec["agg"]["dx_similarity"],
                    "morph_sem": rec["agg"]["morph_sem_0_1"],
                    "morph_syntax_auto": rec["morph_syntax_auto_0_1"],
                    "cross_penalty": rec["agg"]["cross_penalty_0_1"],
                    "text_block": rec["text_block_0_100"],
                    "morph_block": rec["morph_block_0_100"],
                    "final_overall": rec["final_overall"]
                })
            except Exception as e:
                tqdm.write(f"[ERROR] id={_id} -> {e}")
            finally:
                processed += 1
                pbar.update(1)
                if processed % 20 == 0 or processed == len(ids):
                    pbar.set_postfix_str(f"done: {processed}/{len(ids)}")
        pbar.close()


    write_jsonl(per, out_dir / "per_item.jsonl")
    write_csv(rows, out_dir / "summary.csv")
    def mean(key): 
        return round(sum(float(r[key]) for r in rows)/len(rows),4) if rows else 0.0
    macro = {
        "model_alias": model_alias,
        "n_samples": len(rows),
        "mean_reasoning_0_100": mean("reasoning"),
        "mean_dx_0_100": mean("dx"),
        "mean_dx_similarity": mean("dx_similarity"),
        "mean_morph_sem_0_1": mean("morph_sem"),
        "mean_morph_syntax_auto_0_1": mean("morph_syntax_auto"),
        "mean_cross_penalty_0_1": mean("cross_penalty"),
        "mean_text_block_0_100": mean("text_block"),
        "mean_morph_block_0_100": mean("morph_block"),
        "mean_final_overall": mean("final_overall")
    }
    (out_dir / "macro.json").write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")


    meta = {
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "pred_sha256": file_sha256(pred_path),
        "provider": {"name": provider.name, "base_url": provider.url, "model": provider.model},
        "judge_cfg": {
            "n_voters": n_voters,
            "with_image_judge": with_image_judge,
            "image_weight": image_weight,
            "weights": weights
        },
        "timestamp": int(time.time())
    }
    (out_dir / "model_info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tqdm.write(f"[DONE] {model_alias}: n={macro['n_samples']} | mean_final={macro['mean_final_overall']}")

# ------------------- CLI -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model_alias", required=True)
    ap.add_argument("--out_dir", default="./eval_task3_2/results")

    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("API_MODEL", ""))

    ap.add_argument("--n_voters", type=int, default=1)
    ap.add_argument("--max_workers", type=int, default=16)


    ap.add_argument("--with_image_judge", action="store_true")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--image_weight", type=float, default=0.3)

    ap.add_argument("--overwrite", action="store_true")


    ap.add_argument("--w_reasoning", type=float, default=0.33, help="text block: reasoning weight")
    ap.add_argument("--w_diagnosis", type=float, default=0.33, help="text block: diagnosis weight")
    ap.add_argument("--w_morph_semantic", type=float, default=1.0, help="morphology block: semantic score weight; the rest is assigned to syntax")

    ap.add_argument("--w_text", type=float, default=0.66, help="overall score: text block weight")
    ap.add_argument("--w_morph", type=float, default=0.34, help="overall score: morphology block weight")
    ap.add_argument("--w_cross", type=float, default=0.0, help="cross-consistency penalty weight; final score is multiplied by (1 - w_cross * penalty)")

    args = ap.parse_args()

    model_dir = Path(args.out_dir) / args.model_alias
    provider = ApiProvider("Judge", args.base_url, args.api_key, args.model, max_workers=args.max_workers)
    provider.ping()

    weights = {
        "w_reasoning": args.w_reasoning,
        "w_diagnosis": args.w_diagnosis,
        "w_morph_semantic": args.w_morph_semantic,
        "w_text": args.w_text,
        "w_morph": args.w_morph,
        "w_cross": args.w_cross
    }

    s_tm = max(1e-9, weights["w_text"] + weights["w_morph"])
    weights["w_text"]  /= s_tm
    weights["w_morph"] /= s_tm

    s_td = max(1e-9, weights["w_reasoning"] + weights["w_diagnosis"])
    weights["w_reasoning"] /= s_td
    weights["w_diagnosis"] /= s_td

    run_eval(Path(args.gt), Path(args.pred), provider, model_dir, args.model_alias,
             args.n_voters, args.max_workers, args.with_image_judge, Path(args.image_root),
             args.image_weight, args.overwrite, weights)

if __name__ == "__main__":
    main()
