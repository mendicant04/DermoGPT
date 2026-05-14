# coding: utf-8

import argparse, warnings, json, os, sys, torch, math, multiprocessing, csv, re
from tqdm import tqdm
from PIL import Image  # noqa: F401
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# =========================

# =========================

SKINCON_SYS_PROMPT = (
    "You are a dermatology VQA classifier.\n"
    "Output EXACTLY in this order:\n"
    "1) A <morph> block containing ONLY a valid JSON object with EXACTLY one key:\n"
    '   {\n'
    '     "morphological_features_skincon": [ ... ]\n'
    '   }\n'
    "   - The array may be empty or include one or more strings chosen ONLY from this CLOSED SET (case-sensitive, spell exactly):\n"
    '   ["Abscess","Acuminate","Atrophy","Black","Blue","Brown(Hyperpigmentation)","Bulla","Burrow","Comedo","Crust","Cyst","Dome-shaped","Erosion","Erythema","Excoriation","Exophytic/Fungating","Exudate","Fissure","Flat topped","Friable","Gray","Induration","Lichenification","Macule","Nodule","Papule","Patch","Pedunculated","Pigmented","Plaque","Poikiloderma","Purple","Purpura/Petechiae","Pustule","Salmon","Scale","Scar","Sclerosis","Telangiectasia","Translucent","Ulcer","Umbilicated","Vesicle","Warty/Papillomatous","Wheal","White(Hypopigmentation)","Xerosis","Yellow"]\n'
    "   - Include ONLY features visibly present.\n"
    "   - Sort the array alphabetically.\n"
    "2) A blank line, then EXACTLY ONE detailed clinical morphological paragraph (no bullet points, no lists, no diagnosis/management, no probabilities).\n"
    "STRICT RULES: Do NOT add code fences. Do NOT add any extra text before/after the required content. Do NOT include a separate 'detailed_description' key in JSON.\n"
)

DERM7PT_SYS_PROMPT = (
    "You are a dermoscopy VQA classifier.\n"
    "Output EXACTLY in this order:\n"
    "1) A <morph> block containing ONLY a valid JSON object with EXACTLY one key:\n"
    '   {\n'
    '     "morphological_features_Derm7pt": {\n'
    '       "pigment_network": "absent" | "typical" | "atypical",\n'
    '       "blue_whitish_veil": "absent" | "present",\n'
    '       "vascular_structures": "absent" | "arborizing" | "comma" | "hairpin" | "within regression" | "wreath" | "dotted" | "linear irregular",\n'
    '       "pigmentation": "absent" | "diffuse regular" | "localized regular" | "diffuse irregular" | "localized irregular",\n'
    '       "streaks": "absent" | "regular" | "irregular",\n'
    '       "dots_and_globules": "absent" | "regular" | "irregular",\n'
    '       "regression_structures": "absent" | "blue areas" | "white areas" | "combinations"\n'
    '     }\n'
    '   }\n'
    "   - Use the EXACT lowercase_snake_case keys shown above (no extra/missing keys).\n"
    "   - If a structure is not present, set its value to \"absent\".\n"
    "2) A blank line, then EXACTLY ONE detailed dermoscopic paragraph (no bullet points, no lists, no diagnosis/management, no probabilities).\n"
    "STRICT RULES: Do NOT add code fences. Do NOT add any extra text before/after the required content. Do NOT include a separate 'detailed_description' key in JSON.\n"
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

processor = None
model = None
device = "cuda"

def build_conversation_for_model(item, image_abs_path, sys_prompt_text=None):

    human_turn = next((t for t in item.get("conversations", []) if t.get("from")=="human"), None)
    if human_turn is None:
        raise ValueError(f"sample {item.get('id')} is missing a human turn")
    question_text = human_turn.get("value","").replace("<image>", "", 1).lstrip("\n")
    conv = []
    if sys_prompt_text:
        conv.append({"role":"system","content":[{"type":"text","text":sys_prompt_text}]})
    conv.append({
        "role":"user",
        "content":[
            {"type":"image","image":image_abs_path},
            {"type":"text","text":question_text}
        ]
    })
    return conv

MORPH_BLOCK_RE = re.compile(r"<morph>\s*{.*?}\s*</morph>", re.S)

def warn_if_format_off(model_text, derm_mode):
    if not MORPH_BLOCK_RE.search(model_text):
        print("[Warning] No <morph> JSON block was detected, or the format is incomplete.", file=sys.stderr)

    try:

        m = MORPH_BLOCK_RE.search(model_text)
        if m:
            json_str = m.group(0)
            json_str = json_str.replace("<morph>", "").replace("</morph>", "").strip()
            obj = json.loads(json_str)
            if derm_mode:
                if "morphological_features_Derm7pt" not in obj:
                    print("[Warning] Derm7pt JSON is missing morphological_features_Derm7pt.", file=sys.stderr)
                else:
                    inner = obj["morphological_features_Derm7pt"]
                    expect_keys = ["pigment_network","blue_whitish_veil","vascular_structures","pigmentation","streaks","dots_and_globules","regression_structures"]
                    for k in expect_keys:
                        if k not in inner:
                            print(f"[Warning] Derm7pt JSON is missing key: {k}", file=sys.stderr)
            else:
                if "morphological_features_skincon" not in obj:
                    print("[Warning] SkinCon JSON is missing morphological_features_skincon.", file=sys.stderr)
    except Exception:
        print("[Warning] Failed to parse JSON inside <morph>.", file=sys.stderr)

def run_inference_on_item(item, image_base_path, generation_args,
                          idx_full, idx_base, idx_tail2,
                          use_system_prompt=True, enable_warn=True):
    global processor, model, device

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
    if use_system_prompt:
        img_type = resolve_image_type(image_rel, idx_full, idx_base, idx_tail2)
        derm_mode = is_dermoscopy(img_type)
        sys_prompt = DERM7PT_SYS_PROMPT if derm_mode else SKINCON_SYS_PROMPT

    try:
        conv = build_conversation_for_model(item, image_abs, sys_prompt_text=sys_prompt)
        prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conv)
        if not image_inputs:
            print(f"[{device}] Warning: process_vision_info did not load the image (ID:{item.get('id')}), skipping.", file=sys.stderr)
            return None

        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        generation_kwargs = dict(inputs, **generation_args)
        if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token_id is None:
            generation_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        input_token_len = inputs["input_ids"].shape[1]
        resp_tokens = outputs[0][input_token_len:]
        model_text = processor.tokenizer.decode(
            resp_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if enable_warn:
            warn_if_format_off(model_text, derm_mode)


        out_item = {"id": item.get("id"), "image": item.get("image"), "conversations": []}
        human_turn = next((t for t in item.get("conversations", []) if t.get("from")=="human"), None)
        out_item["conversations"].append({"from":"human","value": human_turn.get("value","") if human_turn else ""})
        out_item["conversations"].append({"from":"gpt","value": model_text})
        return out_item

    except Exception as e:
        print(f"[{device}] Critical error: failed while processing ID {item.get('id')}: {e}", file=sys.stderr)
        return None

# =========================

# =========================

def worker_proc(tasks, worker_idx, device_id, image_base_path, generation_args, cli_args,
                shared_jsonl_path, lock, processed_ids, idx_full, idx_base, idx_tail2):
    global processor, model, device
    device = device_id
    print(f"[Worker {worker_idx} on {device}]: Loading model...", flush=True)
    disable_torch_init()
    model_name = get_model_name_from_path(cli_args.model_path)
    use_flash_attn = not cli_args.disable_flash_attention
    processor, model = load_pretrained_model(
        model_base=cli_args.model_base, model_path=cli_args.model_path,
        device_map=device, model_name=model_name,
        load_4bit=cli_args.load_4bit, load_8bit=cli_args.load_8bit,
        device=device, use_flash_attn=use_flash_attn
    )
    model.eval()
    print(f"[Worker {worker_idx} on {device}]: Model ready, {len(tasks)} items.", flush=True)

    for item in tqdm(tasks, desc=f"Worker {worker_idx}", position=worker_idx, file=sys.stdout):
        _id = item.get("id")
        if processed_ids is not None and _id in processed_ids:
            continue
        out_item = run_inference_on_item(
            item, image_base_path, generation_args,
            idx_full, idx_base, idx_tail2,
            use_system_prompt=not cli_args.disable_system_prompt,
            enable_warn=cli_args.warn_format
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
    print(f"System Prompt: {'DISABLED' if args.disable_system_prompt else 'AUTO (<morph> JSON + paragraph)'}")
    print(f"Warn Format: {'ON' if args.warn_format else 'OFF'}")

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

    parser.add_argument("--model-path", type=str, required=True)
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
    parser.add_argument("--json-files", type=str, default="task1_2_short.jsonl", help="comma-separated filenames; leave empty to process all .json/.jsonl files in the directory")


    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-suffix", type=str, default="", help="optional suffix for output filenames")
    parser.add_argument("--output-dir", type=str, default="", help="directory for JSONL/JSON outputs; defaults to --json-dir")
    parser.add_argument("--disable-system-prompt", action="store_true", help="disable the system prompt, useful for fine-tuned models")
    parser.add_argument("--warn-format", action="store_true", help="log warnings only when <morph> is missing or JSON keys are incomplete")
    


    parser.add_argument("--metadata-isic", type=str, default="dataset_final/isic/isic_metadata_with_features.csv")
    parser.add_argument("--metadata-aaa",  type=str, default="dataset_final/aaa_non_isic/combined_dataset_correct_pumch_with_features_multithreaded.csv")

    args = parser.parse_args()
    main(args)
