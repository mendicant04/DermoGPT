# eval_task1_llm_judge_api_single.py
# -*- coding: utf-8 -*-

import os, re, json, csv, time, argparse, concurrent.futures, threading, base64, mimetypes, hashlib
from pathlib import Path
from typing import Dict, Any, List
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

# ------------------- Prompts -------------------
SYSTEM_PROMPT_TEXT = (
    "You are a strict, no-nonsense clinical dermatology evaluator. "
    "You DO NOT see the image; evaluate ONLY by comparing the REFERENCE vs the CANDIDATE text. "
    "Use dermatology morphology standards. Avoid rewarding verbosity; penalize contradictions and invented findings. "
    "Focus on: anatomical site, number/arrangement, primary lesion types, color, shape, borders, surface features, size/extent, "
    "distribution/pattern, and special/contextual features (e.g., pen markings, dermoscopic 7-point structures if applicable). "
    "Return STRICT JSON only."
)

USER_TEMPLATE_TEXT = """[Task Prompt]
{task_prompt}

[REFERENCE]
{reference}

[CANDIDATE]
{candidate}

Evaluate as follows:
1) Decompose REFERENCE into ≤25 atomic CLAIMS.
2) For each CLAIM, label wrt CANDIDATE: Supported, PartiallySupported, Contradicted, Missing, or Vague.
3) Identify any EXTRA INCORRECT statements in CANDIDATE.
4) Score:
   recall_like = (Supported + 0.5*PartiallySupported) / max(1, total_ref_claims)
   precision_penalty = min(1.0, (Contradicted + ExtraIncorrect) / max(1, total_ref_claims))
   overall [0-100] = round(100 * max(0, recall_like - 0.5*precision_penalty), 1)
   Provide rubric sub-scores (accuracy, completeness, consistency) in [0,1].
JSON ONLY. Schema:
{{
  "claims": [{{"text":"...","label":"Supported|PartiallySupported|Contradicted|Missing|Vague"}}],
  "counts": {{"supported":0,"partial":0,"contradicted":0,"missing":0,"vague":0,"extra_incorrect":0,"total_ref_claims":0}},
  "rubric": {{"accuracy":0.0,"completeness":0.0,"consistency":0.0}},
  "overall": 0.0,
  "short_feedback": "≤40 words concise justification"
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

def judge_text(provider: ApiProvider, task_prompt: str, reference: str, candidate: str) -> dict:
    user = USER_TEMPLATE_TEXT.format(task_prompt=task_prompt.strip(), reference=reference.strip(), candidate=candidate.strip())
    return call_chat(provider, [{"role":"system","content":SYSTEM_PROMPT_TEXT},
                                {"role":"user","content":user}],
                     temperature=0.0, max_tokens=8192, json_mode=True)

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

    return call_chat(provider, messages, temperature=0.0, max_tokens=8192, json_mode=False)


def agg_text_votes(vs: List[dict]) -> dict:
    n = max(1, len(vs))
    overall = round(sum(v.get("overall",0.0) for v in vs)/n, 2)
    acc = sum(v.get("rubric",{}).get("accuracy",0.0) for v in vs)/n
    comp= sum(v.get("rubric",{}).get("completeness",0.0) for v in vs)/n
    cons= sum(v.get("rubric",{}).get("consistency",0.0) for v in vs)/n
    keys=["supported","partial","contradicted","missing","vague","extra_incorrect","total_ref_claims"]
    counts={k: round(sum(v.get("counts",{}).get(k,0) for v in vs)/n) for k in keys}
    fb=" | ".join(v.get("short_feedback","") for v in vs if v.get("short_feedback"))[:240]
    return {"overall":overall, "rubric":{"accuracy":round(acc,3),"completeness":round(comp,3),"consistency":round(cons,3)},
            "counts":counts,"short_feedback":fb}

def agg_img_votes(vs: List[dict]) -> dict:
    if not vs: return {"cand_image_penalty":0.0,"ref_image_penalty":0.0,"notes":""}
    n=len(vs)
    cand = sum(float(v.get("cand_image_penalty",0.0)) for v in vs)/n
    ref  = sum(float(v.get("ref_image_penalty",0.0)) for v in vs)/n
    notes=" | ".join(v.get("notes","") for v in vs if v.get("notes"))[:200]
    return {"cand_image_penalty":round(cand,3),"ref_image_penalty":round(ref,3),"notes":notes}


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def run_eval(gt_path: Path, pred_path: Path, provider: ApiProvider,
             out_dir: Path, model_alias: str, n_voters: int, max_workers: int,
             with_image_judge: bool, image_root: Path, image_weight: float,
             overwrite: bool):
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
        reference   = gt_map[_id]["answer"]
        candidate   = pred_map[_id]["answer"]

        # text votes
        tv=[]
        for _ in range(max(1,n_voters)):
            tv.append(judge_text(provider, task_prompt, reference, candidate)); time.sleep(0.05)
        t_agg = agg_text_votes(tv)

        final_overall = t_agg["overall"]
        img_agg=None; img_path=None
        if with_image_judge:
            rel = gt_map[_id].get("image")
            if rel:
                p = Path(rel); img_path = p if p.is_absolute() else (image_root/rel)
                if img_path.exists():
                    iv=[]
                    for _ in range(max(1,n_voters)):
                        iv.append(judge_image(provider, reference, candidate, img_path)); time.sleep(0.05)
                    img_agg = agg_img_votes(iv)
                    final_overall = round(final_overall*(1.0 - image_weight*img_agg["cand_image_penalty"]), 2)
                else:
                    tqdm.write(f"[WARN] image not found: id={_id} -> {img_path}")

        rec = {
            "id": _id,
            "task_prompt": task_prompt,
            "reference": reference,
            "candidate": candidate,
            "text_judge_votes": tv,
            "text_aggregate": t_agg,
            "image_audit": img_agg,
            "final_overall": final_overall,
            "image_path": str(img_path) if img_path else None
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
                    "overall_text": rec["text_aggregate"]["overall"],
                    "final_overall": rec["final_overall"],
                    "accuracy": rec["text_aggregate"]["rubric"]["accuracy"],
                    "completeness": rec["text_aggregate"]["rubric"]["completeness"],
                    "consistency": rec["text_aggregate"]["rubric"]["consistency"],
                    "contradicted": rec["text_aggregate"]["counts"]["contradicted"],
                    "missing": rec["text_aggregate"]["counts"]["missing"],
                    "extra_incorrect": rec["text_aggregate"]["counts"]["extra_incorrect"],
                    "image_penalty": rec["image_audit"]["cand_image_penalty"] if rec["image_audit"] else 0.0
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
        "mean_overall_text": mean("overall_text"),
        "mean_final_overall": mean("final_overall"),
        "mean_accuracy": mean("accuracy"),
        "mean_completeness": mean("completeness"),
        "mean_consistency": mean("consistency"),
        "mean_contradicted": mean("contradicted"),
        "mean_missing": mean("missing"),
        "mean_image_penalty": mean("image_penalty")
    }
    (out_dir / "macro.json").write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")


    meta = {
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "pred_sha256": file_sha256(pred_path),
        "provider": {"name": provider.name, "base_url": provider.url, "model": provider.model},
        "judge_cfg": {"n_voters": n_voters, "with_image_judge": with_image_judge, "image_weight": image_weight},
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
    ap.add_argument("--out_dir", default="./eval_task1/results")
    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", ""))
    ap.add_argument("--api_key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("API_MODEL", ""))
    ap.add_argument("--n_voters", type=int, default=3)
    ap.add_argument("--max_workers", type=int, default=30)
    ap.add_argument("--with_image_judge", action="store_true")
    ap.add_argument("--image_root", default="dataset_final")
    ap.add_argument("--image_weight", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.out_dir) / args.model_alias
    provider = ApiProvider("Judge", args.base_url, args.api_key, args.model, max_workers=args.max_workers)
    provider.ping()

    run_eval(Path(args.gt), Path(args.pred), provider, model_dir, args.model_alias,
             args.n_voters, args.max_workers, args.with_image_judge, Path(args.image_root),
             args.image_weight, args.overwrite)

if __name__ == "__main__":
    main()
