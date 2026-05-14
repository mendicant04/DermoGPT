# coding: utf-8
import os

import argparse, warnings, json, os, sys, math, multiprocessing, csv, re, time, base64, mimetypes, threading
from pathlib import Path

from tqdm import tqdm
from PIL import Image  # noqa: F401
import requests

warnings.filterwarnings("ignore")

# =========================
# Enumerations / Closed sets
# =========================
SKINCON_SET = [
    "Abscess","Acuminate","Atrophy","Black","Blue","Brown(Hyperpigmentation)","Bulla","Burrow","Comedo","Crust","Cyst",
    "Dome-shaped","Erosion","Erythema","Excoriation","Exophytic/Fungating","Exudate","Fissure","Flat topped","Friable",
    "Gray","Induration","Lichenification","Macule","Nodule","Papule","Patch","Pedunculated","Pigmented","Plaque",
    "Poikiloderma","Purple","Purpura/Petechiae","Pustule","Salmon","Scale","Scar","Sclerosis","Telangiectasia",
    "Translucent","Ulcer","Umbilicated","Vesicle","Warty/Papillomatous","Wheal","White(Hypopigmentation)","Xerosis","Yellow"
]

# =========================
# System Prompts (Task 3.2)
# =========================

SKINCON_SYS_PROMPT_T32 = (
    "You are a dermatology VQA assistant.\n"
    "Output EXACTLY THREE blocks in this order and nothing else:\n"
    "1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process.</reasoning>\n"
    "2) <morph>\n"
    "{\n"
    '  "morphological_features_skincon": [ ... ]\n'
    "}\n"
    "</morph>\n"
    "3) <final_diagnosis>ONE label from the provided taxonomy</final_diagnosis>\n"
    "\n"
    "Strict rules:\n"
    "- In <morph>, return ONLY a valid JSON object with EXACTLY one key \"morphological_features_skincon\".\n"
    "- The value is an array (possibly empty) of strings chosen ONLY from this CLOSED set (case-sensitive, spell exactly):\n"
    f"- {SKINCON_SET}\n"
    "- Include ONLY features visibly present; if none, use []. Sort the array alphabetically.\n"
    "- Do NOT add code fences or extra text anywhere.\n"
)

DERM7PT_SYS_PROMPT_T32 = (
    "You are a dermoscopy VQA assistant.\n"
    "Output EXACTLY THREE blocks in this order and nothing else:\n"
    "1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process.</reasoning>\n"
    "2) <morph>\n"
    "{\n"
    '  "morphological_features_Derm7pt": {\n'
    '    "pigment_network": "absent" | "typical" | "atypical",\n'
    '    "blue_whitish_veil": "absent" | "present",\n'
    '    "vascular_structures": "absent" | "arborizing" | "comma" | "hairpin" | "within regression" | "wreath" | "dotted" | "linear irregular",\n'
    '    "pigmentation": "absent" | "diffuse regular" | "localized regular" | "diffuse irregular" | "localized irregular",\n'
    '    "streaks": "absent" | "regular" | "irregular",\n'
    '    "dots_and_globules": "absent" | "regular" | "irregular",\n'
    '    "regression_structures": "absent" | "blue areas" | "white areas" | "combinations"\n'
    "  }\n"
    "}\n"
    "</morph>\n"
    "3) <final_diagnosis>ONE label from the provided taxonomy</final_diagnosis>\n"
    "\n"
    "Strict rules:\n"
    "- Use EXACT lowercase_snake_case keys shown above; output ALL keys (one value per key; no extra/missing keys).\n"
    '- If a structure is not present, set it to "absent".\n'
    "- Do NOT add code fences or extra text anywhere.\n"
)

# =========================

# =========================

def _norm_path(p: str) -> str:
    if p is None: return ""
    p = p.strip().replace("\\", "/")
    while p.startswith("./") or p.startswith("/"):
        p = p[1:]
    return p.lower()

def _tail2_key(norm_path: str) -> str:
    parts = [x for x in norm_path.split("/") if x]
    if not parts: return ""
    return parts[-1] if len(parts) == 1 else "/".join(parts[-2:])

def load_image_type_index(csv_paths):
    full_map, base_to_types, tail2_to_types = {}, {}, {}
    for csv_path in csv_paths:
        if not csv_path or not os.path.exists(csv_path): continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = {k.lower(): k for k in (reader.fieldnames or [])}
                col_rel, col_type = cols.get("relative_path"), cols.get("image_type")
                if not col_rel or not col_type:
                    print(f"[Index] Warning: {csv_path} missing relative_path or image_type", file=sys.stderr)
                    continue
                for row in reader:
                    rp, it = row.get(col_rel), row.get(col_type)
                    if not rp or not it: continue
                    rp_norm = _norm_path(rp)
                    full_map[rp_norm] = it
                    base = os.path.basename(rp_norm)
                    tail2 = _tail2_key(rp_norm)
                    base_to_types.setdefault(base, set()).add(it)
                    tail2_to_types.setdefault(tail2, set()).add(it)
        except Exception as e:
            print(f"[Index] Reading {csv_path} failed: {e}", file=sys.stderr)
    base_map = {k: list(v)[0] for k, v in base_to_types.items() if len(v) == 1}
    tail2_map = {k: list(v)[0] for k, v in tail2_to_types.items() if len(v) == 1}
    print(f"[Index] full_map={len(full_map)} base_map={len(base_map)} tail2_map={len(tail2_map)}")
    return full_map, base_map, tail2_map

def resolve_image_type(image_rel_path, idx_full, idx_base, idx_tail2):
    p_norm = _norm_path(image_rel_path)
    if p_norm in idx_full: return idx_full[p_norm]
    base = os.path.basename(p_norm)
    if base in idx_base: return idx_base[base]
    tail2 = _tail2_key(p_norm)
    if tail2 in idx_tail2: return idx_tail2[tail2]

    low = p_norm
    if "/derm7pt/" in low or "/dermoscopy" in low: return "dermoscopy"
    if "dermnet" in low: return "clinical"
    return "clinical"

def is_dermoscopy(image_type: str) -> bool:
    return "derm" in (image_type or "").lower()

# =========================
# I/O
# =========================

def load_samples(file_path):
    if file_path.lower().endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e) or "line 2" in str(e):
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]
            raise
    if not isinstance(data, list):
        raise ValueError(f"{file_path} top-level JSON value is not an array.")
    return data

def jsonl_ids(path):
    ids = set()
    if not os.path.exists(path): return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: ids.add(json.loads(line).get("id"))
            except Exception: pass
    return ids

def merge_jsonl_to_array(input_items, jsonl_path, out_array_path):
    id2obj = {}
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file does not exist: {jsonl_path}", file=sys.stderr)
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                id2obj[obj.get("id")] = obj
    out, missing = [], 0
    for it in input_items:
        _id = it.get("id")
        if _id in id2obj: out.append(id2obj[_id])
        else: missing += 1
    if missing:
        print(f"Warning: {missing} samples did not produce outputs.", file=sys.stderr)
    with open(out_array_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Merge complete] {out_array_path} total {len(out)} items.")

# =========================

# =========================

ERROR_THRESHOLD = 100
PAUSE_DURATION_SECONDS = 20 * 60
IO_SEMA = threading.Semaphore(4)
device = "api"

class ApiProvider:
    def __init__(self, name, base_url, api_key, model, max_workers=12, timeout=(10, 180)):
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
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def ping(self):
        try:
            r = requests.post(
                self.url,
                headers=self.headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
                timeout=(5, 10),
            )
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
                tqdm.write(
                    f"\n[CRITICAL] '{self.name}' paused {PAUSE_DURATION_SECONDS/60:.0f} min (too many errors)"
                )
                self.consecutive_errors = 0

def b64_image_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "application/octet-stream"
    with IO_SEMA:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def call_chat(provider: ApiProvider, messages: list,
              max_tokens: int = 4096,
              temperature: float = 0.0,
              retry: int = 3) -> str:
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for a in range(retry):
        try:
            if provider.is_paused():
                time.sleep(5)
            r = requests.post(
                provider.url,
                headers=provider.headers(),
                json=payload,
                timeout=provider.timeout,
            )
            r.raise_for_status()
            data = r.json()
            txt = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not txt:
                raise ValueError("Empty response")
            provider.record_success()
            return txt
        except Exception as e:
            provider.record_failure()
            tqdm.write(f"[RETRY {a+1}/{retry}] {provider.name} -> {e}")
            time.sleep(1.0 * (a + 1))
    raise RuntimeError("API call failed after retries")

# =========================

# =========================

RE_BLOCKS = re.compile(
    r"<reasoning>.*?</reasoning>\s*<morph>\s*{.*?}\s*</morph>\s*<final_diagnosis>.*?</final_diagnosis>",
    re.S
)
RE_MORPH_JSON = re.compile(r"<morph>\s*({.*?})\s*</morph>", re.S)

def warn_if_format_off(model_text, derm_mode):
    if not RE_BLOCKS.search(model_text):
        print("[Warning] The three blocks (<reasoning>/<morph>/<final_diagnosis>) are incomplete or out of order.", file=sys.stderr)
    m = RE_MORPH_JSON.search(model_text)
    if not m:
        print("[Warning] No JSON was detected inside <morph>.", file=sys.stderr); return
    try:
        obj = json.loads(m.group(1))
        if derm_mode:
            inner = obj.get("morphological_features_Derm7pt", None)
            expect = ["pigment_network","blue_whitish_veil","vascular_structures","pigmentation","streaks","dots_and_globules","regression_structures"]
            if not isinstance(inner, dict):
                print("[Warning] Derm7pt: morphological_features_Derm7pt is not an object.", file=sys.stderr)
            else:
                miss = [k for k in expect if k not in inner]
                if miss: print(f"[Warning] Derm7pt: missing keys {miss}", file=sys.stderr)
        else:
            arr = obj.get("morphological_features_skincon", None)
            if not isinstance(arr, list):
                print("[Warning] SkinCon: morphological_features_skincon is not an array.", file=sys.stderr)
    except Exception:
        print("[Warning] Failed to parse <morph> JSON.", file=sys.stderr)

# =========================

# =========================

def run_inference_on_item(item, image_base_path, generation_args,
                          idx_full, idx_base, idx_tail2,
                          use_system_prompt=True, enable_warn=True,
                          provider: ApiProvider = None):
    global device

    image_rel = item.get("image")
    if not image_rel:
        print(f"[{device}] Warning: {item.get('id')} missing image field, skipping.", file=sys.stderr)
        return None

    image_abs = os.path.join(image_base_path, image_rel)
    if not os.path.exists(image_abs):
        print(f"[{device}] Warning: image does not exist (ID:{item.get('id')}) at {image_abs}, skipping.", file=sys.stderr)
        return None


    sys_prompt = None
    derm_mode = False
    img_type = resolve_image_type(image_rel, idx_full, idx_base, idx_tail2)
    derm_mode = is_dermoscopy(img_type)
    if use_system_prompt:
        sys_prompt = DERM7PT_SYS_PROMPT_T32 if derm_mode else SKINCON_SYS_PROMPT_T32


    human_turn = next((t for t in item.get("conversations", []) if t.get("from") == "human"), None)
    if human_turn is None:
        print(f"[{device}] Warning: sample {item.get('id')} missing human turn, skipping.", file=sys.stderr)
        return None
    question_text = human_turn.get("value", "").replace("<image>", "", 1).lstrip("\n")


    img_url = b64_image_url(Path(image_abs))
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question_text},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    })

    max_new_tokens = int(generation_args.get("max_new_tokens", 4096))
    temperature = float(generation_args.get("temperature", 0.0))

    try:
        model_text = call_chat(
            provider,
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        if enable_warn:
            warn_if_format_off(model_text, derm_mode)


        out_item = {
            "id": item.get("id"),
            "image": item.get("image"),
            "conversations": []
        }
        out_item["conversations"].append({
            "from": "human",
            "value": human_turn.get("value", "") if human_turn else ""
        })
        out_item["conversations"].append({
            "from": "gpt",
            "value": model_text
        })
        return out_item

    except Exception as e:
        print(f"[{device}] Critical error while processing ID {item.get('id')}: {e}", file=sys.stderr)
        return None

# =========================

# =========================

def worker_proc(tasks, worker_idx, device_id, image_base_path, generation_args, cli_args,
                shared_jsonl_path, lock, processed_ids, idx_full, idx_base, idx_tail2):
    global device
    device = f"api-worker-{worker_idx}"
    print(f"[Worker {worker_idx} on {device}]: Initializing API provider...", flush=True)

    provider = ApiProvider(
        name=f"Infer-{worker_idx}",
        base_url=cli_args.base_url,
        api_key=cli_args.api_key,
        model=cli_args.model,
        max_workers=1,
        timeout=(10, 180),
    )
    # provider.ping()
    print(f"[Worker {worker_idx} on {device}]: Provider ready, {len(tasks)} items.", flush=True)

    for item in tqdm(tasks, desc=f"Worker {worker_idx}", position=worker_idx, file=sys.stdout):
        _id = item.get("id")
        if processed_ids is not None and _id in processed_ids:
            continue
        out_item = run_inference_on_item(
            item, image_base_path, generation_args,
            idx_full, idx_base, idx_tail2,
            use_system_prompt=not cli_args.disable_system_prompt,
            enable_warn=cli_args.warn_format,
            provider=provider
        )
        if out_item is None:
            continue
        with lock:
            with open(shared_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

def main(args):
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    output_dir = args.output_dir or args.json_dir
    os.makedirs(output_dir, exist_ok=True)

    idx_full, idx_base, idx_tail2 = load_image_type_index([args.metadata_isic, args.metadata_aaa])

    if args.json_files:
        files = [x.strip() for x in args.json_files.split(",") if x.strip()]
    else:
        files = [fn for fn in os.listdir(args.json_dir) if fn.endswith(".json") or fn.endswith(".jsonl")]
        files.sort()

    devices = [args.device for _ in range(args.num_workers)]
    print(f"Image root: {args.image_base_path}")
    print(f"Input directory:   {args.json_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Parallel workers:     {args.num_workers} @ {args.device}")
    print(f"System Prompt: {'DISABLED' if args.disable_system_prompt else 'AUTO (Task3.2format)'}")
    print(f"Warn Format: {'ON' if args.warn_format else 'OFF'}")
    print(f"API base:   {args.base_url}")
    print(f"API model:  {args.model}")

    for json_file in files:
        in_path = os.path.join(args.json_dir, json_file)
        try:
            input_data = load_samples(in_path)
        except Exception as e:
            print(f"\nError: Failed to load {in_path}: {e}, skipping.", file=sys.stderr)
            continue

        base = os.path.splitext(json_file)[0]
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        shared_jsonl = os.path.join(output_dir, f"{base}{suffix}.jsonl")
        out_array   = os.path.join(output_dir, f"{base}{suffix}_infer.json")

        processed_ids = jsonl_ids(shared_jsonl) if os.path.exists(shared_jsonl) else set()
        if processed_ids:
            print(f"[{json_file}] Existing JSONL found; skipped {len(processed_ids)} items.")

        chunk_size = math.ceil(len(input_data) / max(1, args.num_workers))
        chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]

        manager = multiprocessing.Manager()
        lock = manager.Lock()

        job_args = []
        for i in range(len(chunks)):
            job_args.append((
                chunks[i], i, devices[i],
                args.image_base_path, generation_args, args,
                shared_jsonl, lock, processed_ids,
                idx_full, idx_base, idx_tail2
            ))

        print(f"\n--- Processing {json_file}({len(input_data)} items)-> {shared_jsonl}")
        print("\n" * len(chunks))
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            pool.starmap(worker_proc, job_args)

        merge_jsonl_to_array(input_data, shared_jsonl, out_array)
        print(f"--- Done: {json_file}\n")

    print("All files have been processed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")


    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)


    parser.add_argument("--image-base-path", type=str, default="dataset_final", help="dataset root used to resolve relative image paths")
    parser.add_argument("--json-dir", type=str, default="dataset_final/sft_train_test/test/task1_3_benchmark_json", help="directory containing input and output JSON files")
    parser.add_argument("--json-files", type=str, default="task3_2_short.jsonl", help="comma-separated filenames; leave empty to process all .json/.jsonl files in the directory")


    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-suffix", type=str, default="", help="optional suffix for output filenames")
    parser.add_argument("--output-dir", type=str, default="", help="directory for JSONL/JSON outputs; defaults to --json-dir")
    parser.add_argument("--disable-system-prompt", action="store_true", help="disable the system prompt, useful for fine-tuned models")
    parser.add_argument("--warn-format", action="store_true", help="log warnings only when <morph> is missing or JSON keys are incomplete")


    parser.add_argument("--metadata-isic", type=str, default="dataset_final/isic/isic_metadata_with_features.csv")
    parser.add_argument("--metadata-aaa",  type=str, default="dataset_final/aaa_non_isic/combined_dataset_correct_pumch_with_features_multithreaded.csv")


    parser.add_argument("--base_url", type=str, default=os.environ.get("BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model", type=str, default=os.environ.get("API_MODEL", ""))

    args = parser.parse_args()
    main(args)
