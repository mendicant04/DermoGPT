# eval_task1_2_llm_judge_api_single.py
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



SYSTEM_PROMPT_TASK12 = (
    "You are a strict dermatology evaluator for Task 1.2 (morph content + narrative). "
    "You DO NOT see the image. Focus on CONTENT, not formatting. "
    "Both REFERENCE and CANDIDATE may or may not wrap the morph JSON in <morph> tags. "
    "Do NOT penalize missing tags, extra whitespace, or minor ordering/format differences. "
    "If a JSON block is present anywhere, treat the FIRST JSON object as the morph content. "
    "If no JSON is present, infer the morph feature set from the surrounding text. "
    "Schemas you may encounter:\n"
    "  • SkinCon: {\"morphological_features_skincon\": [<feature strings>]} \n"
    "  • Derm7pt: {\"morphological_features_Derm7pt\": {pigment_network, blue_whitish_veil, vascular_structures, pigmentation, streaks, dots_and_globules, regression_structures}} \n"
    "For the narrative comparison, use dermatology morphology standards (site, number/arrangement, primary lesion types, color, shape, borders, surface features, size/extent, distribution/pattern, special/context). "
    "Also check CROSS-CONSISTENCY between the CANDIDATE morph content and CANDIDATE narrative. "
    "Return STRICT JSON only."
)


USER_TEMPLATE_TASK12 = """You will be given REFERENCE and CANDIDATE texts. 
Each may contain a morph JSON (SkinCon or Derm7pt) with or without <morph> tags, 
possibly followed by a narrative paragraph. Do NOT penalize formatting. 
Rules:
- If a JSON object appears anywhere, treat the FIRST JSON object as the morph content.
- If no JSON is found, infer the morph feature set from the surrounding text (best-effort).
- Use synonyms tolerance for semantic matching.

[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Your tasks:
1) MORPH SEMANTICS (content-first): Compare CANDIDATE-morph vs REFERENCE-morph semantically (synonyms allowed). 
   Count supported/missing/contradicted/extra and give a semantic score in [0,1]. 
   If CANDIDATE has no explicit JSON, infer its morph set from the candidate text.

2) TEXT (NARRATIVE): Compare REFERENCE-narrative vs CANDIDATE-narrative using morphology standards. 
   Extract <=25 atomic claims from the REFERENCE-narrative; for each, label CANDIDATE as Supported/PartiallySupported/Contradicted/Missing/Vague. 
   Provide rubric sub-scores (accuracy, completeness, consistency) in [0,1] and overall [0,100] using:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   overall = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)

3) CROSS-CONSISTENCY: Judge if the CANDIDATE narrative contradicts the CANDIDATE morph content. 
   Output a penalty in [0,1] (0=no issue, 1=severe) and short notes.

Output STRICT JSON:
{{
  "morph_semantic": {{
    "schema": "SkinCon" | "Derm7pt" | "Unknown",
    "supported": 0, "missing": 0, "contradicted": 0, "extra": 0,
    "score_semantic": 0.0,
    "notes": "≤60 words"
  }},
  "text_judge": {{
    "claims": [{{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}}],
    "counts": {{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},
    "rubric": {{"accuracy":0.0,"completeness":0.0,"consistency":0.0}},
    "overall": 0.0,
    "short_feedback": "≤40 words"
  }},
  "cross_consistency": {{"penalty": 0.0, "notes": "≤40 words"}}
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
{
  "cand_image_penalty": 0.0,
  "ref_image_penalty": 0.0,
  "notes": "≤60 words describing the key contradictions (if any)"
}
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


TAG_MORPH = re.compile(r"<morph>(.*?)</morph>", re.S | re.I)

def extract_morph_json_blocks(s: str) -> Tuple[str, dict]:
    m = TAG_MORPH.search(s or "")
    if not m:
        return ("", None)
    block = m.group(1)

    j = re.search(r"\{.*\}", block, flags=re.S)
    if not j:
        return (block.strip(), None)
    raw = j.group(0)
    try:
        obj = json.loads(raw)
        return (raw, obj)
    except Exception:
        return (raw, None)

def norm_str(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "")).strip().lower()

def syntax_metrics(gt_obj: dict, ca_obj: dict) -> dict:
    schema = "Unknown"
    json_valid_reference = gt_obj is not None
    json_valid_candidate = ca_obj is not None

    if not (gt_obj and ca_obj):
        return {
            "schema": schema,
            "json_valid_reference": json_valid_reference,
            "json_valid_candidate": json_valid_candidate,
            "syntactic_precision": 0.0,
            "syntactic_recall": 0.0,
            "syntactic_f1": 0.0,
            "syntactic_accuracy": 0.0
        }

    if "morphological_features_skincon" in gt_obj:
        schema = "SkinCon"
        gt_list = gt_obj.get("morphological_features_skincon", [])
        ca_list = ca_obj.get("morphological_features_skincon", [])
        gt_set = set(norm_str(x) for x in (gt_list or []))
        ca_set = set(norm_str(x) for x in (ca_list or []))
        tp = len(gt_set & ca_set)
        prec = tp / max(1, len(ca_set))
        rec  = tp / max(1, len(gt_set))
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        return {
            "schema": schema,
            "json_valid_reference": True,
            "json_valid_candidate": True,
            "syntactic_precision": round(prec,4),
            "syntactic_recall": round(rec,4),
            "syntactic_f1": round(f1,4),
            "syntactic_accuracy": 0.0
        }

    if "morphological_features_Derm7pt" in gt_obj:
        schema = "Derm7pt"
        keys = ["pigment_network","blue_whitish_veil","vascular_structures",
                "pigmentation","streaks","dots_and_globules","regression_structures"]
        gt_kv = gt_obj.get("morphological_features_Derm7pt", {})
        ca_kv = (ca_obj or {}).get("morphological_features_Derm7pt", {})
        total = len(keys); correct = 0
        for k in keys:
            if norm_str(gt_kv.get(k)) == norm_str(ca_kv.get(k)):
                correct += 1
        acc = correct / max(1, total)
        return {
            "schema": schema,
            "json_valid_reference": True,
            "json_valid_candidate": True,
            "syntactic_precision": 0.0, "syntactic_recall": 0.0, "syntactic_f1": 0.0,
            "syntactic_accuracy": round(acc,4)
        }

    # Unknown schema
    return {
        "schema": schema,
        "json_valid_reference": True,
        "json_valid_candidate": True,
        "syntactic_precision": 0.0, "syntactic_recall": 0.0, "syntactic_f1": 0.0, "syntactic_accuracy": 0.0
    }

# ------------------- API Call -------------------
def _extract_json(txt: str) -> dict:
    cleaned = txt.strip().replace("```json","").replace("```","").strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not m: raise ValueError("No JSON object found")
    return json.loads(m.group(0))

def call_chat(provider: ApiProvider, messages: list, max_tokens=8192, temperature=0.0, json_mode=True, retry=3):
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


def judge_task12(provider: ApiProvider, task_prompt: str, reference: str, candidate: str) -> dict:
    user = USER_TEMPLATE_TASK12.format(task_prompt=task_prompt.strip(),
                                       reference=reference.strip(),
                                       candidate=candidate.strip())
    return call_chat(
        provider,
        [{"role":"system","content":SYSTEM_PROMPT_TASK12},
         {"role":"user","content":user}],
        temperature=0.0, max_tokens=8192, json_mode=True
    )


def b64_image_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime: mime = "application/octet-stream"
    with IO_SEMA:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

SYSTEM_PROMPT_IMAGE = SYSTEM_PROMPT_IMAGE
USER_TEMPLATE_IMAGE = USER_TEMPLATE_IMAGE

def judge_image(provider: ApiProvider, reference: str, candidate: str, image_path: Path) -> dict:
    user_text = USER_TEMPLATE_IMAGE.format(reference=reference.strip(), candidate=candidate.strip())
    url = b64_image_url(image_path)
    messages = [{"role":"system","content":SYSTEM_PROMPT_IMAGE},
                {"role":"user","content":[{"type":"text","text":user_text},
                                          {"type":"image_url","image_url":{"url":url}}]}]
    return call_chat(provider, messages, temperature=0.0, max_tokens=8192, json_mode=False)


def agg_text_votes(vs: List[dict]) -> dict:

    if not vs: return {"overall":0.0,"rubric":{"accuracy":0,"completeness":0,"consistency":0},
                       "counts":{},"short_feedback":""}
    n=len(vs)
    over = round(sum(v["text_judge"].get("overall",0.0) for v in vs)/n, 2)
    acc = sum(v["text_judge"].get("rubric",{}).get("accuracy",0.0) for v in vs)/n
    comp= sum(v["text_judge"].get("rubric",{}).get("completeness",0.0) for v in vs)/n
    cons= sum(v["text_judge"].get("rubric",{}).get("consistency",0.0) for v in vs)/n
    fb  = " | ".join(v["text_judge"].get("short_feedback","") for v in vs if v.get("text_judge"))[:240]

    keys=["supported","partial","contradicted","missing","vague","extra_incorrect","total_ref_claims"]
    counts={k: round(sum(v["text_judge"].get("counts",{}).get(k,0) for v in vs)/n) for k in keys}
    return {"overall": over, "rubric":{"accuracy":round(acc,3),"completeness":round(comp,3),"consistency":round(cons,3)},
            "counts": counts, "short_feedback": fb}

def agg_morph_semantic_votes(vs: List[dict]) -> dict:
    if not vs: return {"schema":"Unknown","score_semantic":0.0,"supported":0,"missing":0,"contradicted":0,"extra":0,"notes":""}
    n=len(vs)
    schema_vote=[v["morph_semantic"].get("schema","Unknown") for v in vs]

    schema = max(set(schema_vote), key=schema_vote.count) if schema_vote else "Unknown"
    score = sum(v["morph_semantic"].get("score_semantic",0.0) for v in vs)/n
    sup = round(sum(v["morph_semantic"].get("supported",0) for v in vs)/n)
    mis = round(sum(v["morph_semantic"].get("missing",0) for v in vs)/n)
    con = round(sum(v["morph_semantic"].get("contradicted",0) for v in vs)/n)
    ext = round(sum(v["morph_semantic"].get("extra",0) for v in vs)/n)
    notes=" | ".join(v["morph_semantic"].get("notes","") for v in vs if v.get("morph_semantic"))[:200]
    return {"schema":schema, "score_semantic":round(score,4), "supported":sup, "missing":mis, "contradicted":con, "extra":ext, "notes":notes}

def agg_cross_consistency_votes(vs: List[dict]) -> dict:
    if not vs: return {"penalty":0.0,"notes":""}
    n=len(vs)
    pen = sum(v["cross_consistency"].get("penalty",0.0) for v in vs)/n
    notes=" | ".join(v["cross_consistency"].get("notes","") for v in vs if v.get("cross_consistency"))[:160]
    return {"penalty": round(pen,4), "notes": notes}

def combine_scores(text_overall_0_100: float,
                   morph_semantic_0_1: float,
                   morph_syntactic_auto_0_1: float,
                   cross_penalty_0_1: float,
                   weights: dict) -> Tuple[float, float]:
    ws  = weights.get("morph_semantic_weight", 1.0)
    tw  = weights.get("text_weight", 0.5)
    mw  = weights.get("morph_weight", 0.5)
    ccw = weights.get("cc_weight", 0.0)
    morph_score = 100.0 * max(0.0, min(1.0, ws*morph_semantic_0_1 + (1-ws)*morph_syntactic_auto_0_1))
    final_pre = tw*text_overall_0_100 + mw*morph_score
    final = final_pre * (1.0 - ccw * max(0.0, min(1.0, cross_penalty_0_1)))
    return (round(morph_score,2), round(final,2))


def run_eval(gt_path: Path, pred_path: Path, provider: ApiProvider,
             out_dir: Path, model_alias: str, n_voters: int, max_workers: int,
             with_image_judge: bool, image_root: Path, image_weight: float,
             overwrite: bool,
             weights: dict):
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


        ref_raw, ref_morph = extract_morph_json_blocks(ref_full)
        cand_raw, cand_morph = extract_morph_json_blocks(cand_full)
        syntax = syntax_metrics(ref_morph, cand_morph)

        auto_01 = 0.0
        if syntax["schema"] == "SkinCon":
            auto_01 = syntax.get("syntactic_f1", 0.0)
        elif syntax["schema"] == "Derm7pt":
            auto_01 = syntax.get("syntactic_accuracy", 0.0)
        else:
            auto_01 = max(syntax.get("syntactic_f1",0.0), syntax.get("syntactic_accuracy",0.0))


        votes=[]
        for _ in range(max(1, n_voters)):
            res = judge_task12(provider, task_prompt, ref_full, cand_full)
            votes.append(res); time.sleep(0.05)

        text_agg   = agg_text_votes(votes)
        morph_sem  = agg_morph_semantic_votes(votes)
        cross_agg  = agg_cross_consistency_votes(votes)


        morph_score, final_overall = combine_scores(
            text_overall_0_100 = text_agg["overall"],
            morph_semantic_0_1 = morph_sem.get("score_semantic", 0.0),
            morph_syntactic_auto_0_1 = auto_01,
            cross_penalty_0_1 = cross_agg.get("penalty", 0.0),
            weights = weights
        )


        img_audit = None; image_path=None
        if with_image_judge:
            rel_img = gt_map[_id].get("image")
            if rel_img:
                p = Path(rel_img)
                image_path = p if p.is_absolute() else (image_root/rel_img)
                if image_path.exists():
                    iv=[]
                    for _ in range(max(1,n_voters)):
                        iv.append(judge_image(provider, ref_full, cand_full, image_path)); time.sleep(0.05)
                    cand_pen = sum(float(v.get("cand_image_penalty",0.0)) for v in iv)/max(1,len(iv))
                    img_audit = {"cand_image_penalty": round(cand_pen,3)}
                    final_overall = round(final_overall * (1.0 - image_weight * cand_pen), 2)
                else:
                    tqdm.write(f"[WARN] image not found: id={_id} -> {image_path}")

        rec = {
            "id": _id,
            "task_prompt": task_prompt,
            "reference": ref_full,
            "candidate": cand_full,


            "morph_syntax": syntax,
            "morph_auto_score_0_1": auto_01,        # 0~1


            "votes": votes,
            "text_aggregate": text_agg,             # overall 0~100
            "morph_semantic": morph_sem,            # score_semantic 0~1
            "cross_consistency": cross_agg,         # penalty 0~1


            "morph_score_0_100": morph_score,
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
                    "text_overall": rec["text_aggregate"]["overall"],
                    "morph_auto": rec["morph_auto_score_0_1"],
                    "morph_sem": rec["morph_semantic"]["score_semantic"],
                    "morph_score": rec["morph_score_0_100"],
                    "cross_penalty": rec["cross_consistency"]["penalty"],
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
        "mean_text_overall": mean("text_overall"),
        "mean_morph_auto_0_1": mean("morph_auto"),
        "mean_morph_sem_0_1": mean("morph_sem"),
        "mean_morph_score_0_100": mean("morph_score"),
        "mean_cross_penalty_0_1": mean("cross_penalty"),
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
    ap.add_argument("--out_dir", default="./eval_task1_2/results")

    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("API_MODEL", ""))

    ap.add_argument("--n_voters", type=int, default=3)
    ap.add_argument("--max_workers", type=int, default=16)


    ap.add_argument("--with_image_judge", action="store_true")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--image_weight", type=float, default=0.3)

    ap.add_argument("--overwrite", action="store_true")


    ap.add_argument("--text_weight", type=float, default=0.5)
    ap.add_argument("--morph_weight", type=float, default=0.5)
    ap.add_argument("--morph_semantic_weight", type=float, default=1.0)
    ap.add_argument("--cc_weight", type=float, default=0.0)

    args = ap.parse_args()

    model_dir = Path(args.out_dir) / args.model_alias
    provider = ApiProvider("Judge", args.base_url, args.api_key, args.model, max_workers=args.max_workers)
    provider.ping()

    weights = {
        "text_weight": args.text_weight,
        "morph_weight": args.morph_weight,
        "morph_semantic_weight": args.morph_semantic_weight,
        "cc_weight": args.cc_weight
    }

    s = max(1e-9, weights["text_weight"] + weights["morph_weight"])
    weights["text_weight"] /= s
    weights["morph_weight"] /= s

    run_eval(Path(args.gt), Path(args.pred), provider, model_dir, args.model_alias,
             args.n_voters, args.max_workers, args.with_image_judge, Path(args.image_root),
             args.image_weight, args.overwrite, weights)

if __name__ == "__main__":
    main()
