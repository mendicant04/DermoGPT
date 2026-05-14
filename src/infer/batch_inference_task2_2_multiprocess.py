# coding: utf-8

import argparse, warnings, json, os, sys, torch, math, multiprocessing, re, hashlib, random, statistics
from tqdm import tqdm
from PIL import Image  # noqa: F401
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# =========================

# =========================
TOP_LEVEL_PROMPTS_GEN = [
    "Based on the clinical image, identify the most fitting major dermatological category from the following list: {options_list}.",
    "Observe the skin lesion. Which of these high-level classifications best describes it? Here are the possibilities: {options_list}.",
    "Please provide a broad categorization for the skin condition shown. Your answer should be one of the following: {options_list}."
]
SUB_LEVEL_PROMPTS_GEN = [
    "Correct, the condition is a form of '{parent_category}'. Now, specify the sub-category from this list: {options_list}.",
    "Proceeding from '{parent_category}', which of the following groups does this lesion belong to? {options_list}.",
    "Understood. Let's refine the diagnosis within '{parent_category}'. Please choose the most accurate description from the following: {options_list}."
]
FINAL_LEVEL_PROMPTS_GEN = [
    "We've classified this under '{parent_category}'. Now, provide the definitive diagnosis from the choices available: {options_list}.",
    "Excellent. To finalize, please state the specific diagnosis for '{parent_category}', which should be one of the following: {options_list}.",
    "Perfect. Based on our hierarchical classification ending with '{parent_category}', please identify the definitive diagnosis from this list: {options_list}."
]


HUMAN_CORRECTION_PROMPTS = [
    "Actually, that's incorrect. A closer look reveals features more consistent with '{correct_choice}'. Please correct the path.",
    "That's not quite right. The correct category here should be '{correct_choice}'. Let's proceed with that.",
    "Incorrect. The diagnosis should be '{correct_choice}'. Continue from this category."
]

# =========================

# =========================
def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file top-level value must be a JSON array")
    return data

def jsonl_ids(path):
    ids = set()
    if not os.path.exists(path): return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line).get("id"))
                except Exception:
                    pass
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
        if _id in id2obj:
            out.append(id2obj[_id])
        else:
            missing += 1
    if missing:
        print(f"Warning: {missing} samples did not produce outputs (missing images or errors skipped).", file=sys.stderr)
    with open(out_array_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Merge complete] {out_array_path} total {len(out)} items.")

# =========================

# =========================
def get_options_for_path(hierarchy_top_down, path=None):
    if path is None: path = []
    node = hierarchy_top_down
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return [], []
    if isinstance(node, dict):
        sub_categories = [k for k in node.keys() if k != "_items_"]
        items = node.get("_items_", [])
        return sub_categories, items
    elif isinstance(node, list):
        return [], node
    return [], []

def stable_shuffle(options, seed_str):
    rnd = random.Random(int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (10**9+7))
    opts = list(options)
    rnd.shuffle(opts)
    return opts

def format_options(options):
    return ", ".join(f"\"{opt}\"" for opt in options)

def build_question(level_idx, parent_category, options, sample_id):

    options_shuffled = stable_shuffle(options, f"{sample_id}-{level_idx}")
    if level_idx >= 0 and parent_category is None and options and isinstance(options, list) and any(isinstance(k, str) for k in options):

        tmpl = stable_shuffle(TOP_LEVEL_PROMPTS_GEN, f"top-{sample_id}")[0]
        return tmpl.format(options_list=format_options(options_shuffled)), options_shuffled
    else:


        tmpl_pool = FINAL_LEVEL_PROMPTS_GEN if parent_category and options and not any(o in options for o in ["_items_"]) and isinstance(options, list) and len(options) > 0 and '"' in format_options(options_shuffled) else SUB_LEVEL_PROMPTS_GEN
        tmpl = stable_shuffle(tmpl_pool, f"sub-{sample_id}-{level_idx}")[0]
        return tmpl.format(parent_category=parent_category, options_list=format_options(options_shuffled)), options_shuffled

def parse_gt_from_singleturn(gt_text):

    text = gt_text.strip()
    text = re.sub(r'^[A-Ha-h]\)\s*', '', text)
    text = text.strip().strip('"').strip("'")
    return text

def normalize(s):
    return re.sub(r'\s+', ' ', (s or '')).strip().lower()

def extract_choice_from_text(model_text, options):

    mt = model_text or ""
    mt_norm = normalize(mt)
    idx_hits = []
    for i, opt in enumerate(options):
        on = normalize(opt)
        if on and on in mt_norm:
            idx_hits.append((i, mt_norm.find(on)))
    if idx_hits:

        idx_hits.sort(key=lambda x: x[1])
        return options[idx_hits[0][0]]

    def tokens(x): return set(re.findall(r'\w+', normalize(x)))
    mt_tok = tokens(mt)
    best_i, best_score = -1, 0.0
    for i, opt in enumerate(options):
        t = tokens(opt)
        if not t: continue
        inter = len(mt_tok & t); uni = len(mt_tok | t)
        score = inter / (uni + 1e-9)
        if score > best_score:
            best_score, best_i = score, i
    if best_score >= 0.5:
        return options[best_i]
    return None

# =========================

# =========================
processor = None
model = None
device = "cuda"

def run_one_generation(conversation, generation_args):
    global processor, model, device
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
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
    return model_text

# =========================

# =========================
def run_dynamic_eval_on_item(item, image_base_path, generation_args,
                             mapping_data, hierarchy_bottom_up, hierarchy_top_down,
                             attach_system_prompt=False, max_depth=12):
    global processor, model, device

    image_rel = item.get("image")
    if not image_rel:
        return None, 0, 0, 0
    image_abs = os.path.join(image_base_path, image_rel)
    if not os.path.exists(image_abs):
        print(f"[{device}] Warning: image does not exist (ID:{item.get('id')}) at {image_abs}, skipping.", file=sys.stderr)
        return None, 0, 0, 0


    gt_conv = item.get("conversations", [])
    gt_label = None
    for t in gt_conv:
        if t.get("from") == "gpt":
            gt_label = parse_gt_from_singleturn(t.get("value",""))
            break
    if not gt_label:
        return None, 0, 0, 0
    canonical_dx = mapping_data.get(gt_label, gt_label)
    true_path = hierarchy_bottom_up.get(canonical_dx)
    if not true_path or not isinstance(true_path, list):
        return None, 0, 0, 0


    out_item = {"id": item.get("id"), "image": item.get("image"), "conversations": []}


    convo_for_model = []
    if attach_system_prompt:
        convo_for_model.append({
            "role": "system",
            "content": [{"type": "text", "text": "You are a dermatology VQA assistant."}]
        })


    step_correct = 0
    step_total = 0
    corrections_count = 0

    current_path = []
    depth = 0
    first_turn = True
    parent_category = None

    while depth < max_depth:
        sub_categories, items = get_options_for_path(hierarchy_top_down, current_path)


        ask_items_layer = False
        options_pool = []
        if sub_categories:
            options_pool.extend(sub_categories)
        if (not sub_categories) and items:
            ask_items_layer = True
            options_pool.extend(items)
        if not options_pool:
            break


        gold_label = canonical_dx if ask_items_layer else (true_path[depth] if depth < len(true_path) else None)


        q_text, options_ordered = build_question(
            level_idx=depth if not ask_items_layer else len(true_path),
            parent_category=parent_category,
            options=options_pool,
            sample_id=item.get("id","")
        )


        human_value_to_write = (("<image>\n" + q_text) if first_turn else q_text)
        out_item["conversations"].append({"from": "human", "value": human_value_to_write})


        if first_turn:
            convo_for_model.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_abs},
                    {"type": "text", "text": q_text}
                ]
            })
        else:
            convo_for_model.append({
                "role": "user",
                "content": [{"type": "text", "text": q_text}]
            })

        model_text = run_one_generation(convo_for_model, generation_args)
        if model_text is None:
            out_item["conversations"].append({"from": "gpt", "value": ""})
            break
        out_item["conversations"].append({"from": "gpt", "value": model_text})


        pred_choice = extract_choice_from_text(model_text, options_ordered)
        step_total += 1
        if pred_choice is not None and gold_label is not None and normalize(pred_choice) == normalize(gold_label):
            step_correct += 1

            parent_category = pred_choice
            if ask_items_layer:
                break
            current_path.append(pred_choice)
            depth += 1
            first_turn = False
            continue


        corrections_count += 1
        corr_tmpl = stable_shuffle(HUMAN_CORRECTION_PROMPTS, f"corr-{item.get('id','')}-{depth}")[0]
        corr_text = corr_tmpl.format(correct_choice=gold_label)


        out_item["conversations"].append({"from": "human", "value": corr_text})

        convo_for_model.append({"role": "user", "content": [{"type": "text", "text": corr_text}]})
        corr_reply = run_one_generation(convo_for_model, generation_args)
        out_item["conversations"].append({"from": "gpt", "value": corr_reply or ""})


        parent_category = gold_label
        if ask_items_layer:

            break
        else:
            current_path.append(gold_label)
            depth += 1
            first_turn = False


    out_item["corrections"] = corrections_count
    return out_item, step_correct, step_total, corrections_count

# =========================

# =========================
def worker_proc(tasks, worker_idx, device_id, image_base_path, generation_args, cli_args,
                shared_jsonl_path, lock, processed_ids,
                mapping_data, hierarchy_bottom_up, hierarchy_top_down):
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

    sum_correct, sum_total, sum_corr = 0, 0, 0
    for item in tqdm(tasks, desc=f"Worker {worker_idx}", position=worker_idx, file=sys.stdout):
        _id = item.get("id")
        if processed_ids is not None and _id in processed_ids:
            continue

        out_item, c, t, corr = run_dynamic_eval_on_item(
            item, image_base_path, generation_args,
            mapping_data, hierarchy_bottom_up, hierarchy_top_down,
            attach_system_prompt=not cli_args.disable_system_prompt,
            max_depth=cli_args.max_depth
        )
        sum_correct += c
        sum_total += t
        sum_corr += corr
        if out_item is None:
            continue


        with lock:
            with open(shared_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    return sum_correct, sum_total, sum_corr, len(tasks)

def main(args):
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }


    output_dir = args.output_dir or os.path.dirname(args.json_file) or "."
    os.makedirs(output_dir, exist_ok=True)


    input_data = load_json_array(args.json_file)
    base = os.path.splitext(os.path.basename(args.json_file))[0]
    shared_jsonl = os.path.join(output_dir, f"{base}_mt_eval.jsonl")
    out_array   = os.path.join(output_dir, f"{base}_mt_eval_infer.json")
    metric_path = os.path.join(output_dir, f"{base}_metrics.json")


    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    with open(args.hierarchy_bottom_up, "r", encoding="utf-8") as f:
        hierarchy_bottom_up = json.load(f)
    with open(args.hierarchy_top_down, "r", encoding="utf-8") as f:
        hierarchy_top_down = json.load(f)

    processed_ids = jsonl_ids(shared_jsonl) if os.path.exists(shared_jsonl) else set()
    if processed_ids:
        print(f"[{base}] Existing JSONL found; skipped {len(processed_ids)} items.")


    chunk_size = math.ceil(len(input_data) / max(1, args.num_workers))
    chunks = [input_data[i:i+chunk_size] for i in range(0, len(input_data), chunk_size)]

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    job_args = []
    devices = [args.device for _ in range(len(chunks))]
    for i in range(len(chunks)):
        job_args.append((
            chunks[i], i, devices[i],
            args.image_base_path, generation_args, args,
            shared_jsonl, lock, processed_ids,
            mapping_data, hierarchy_bottom_up, hierarchy_top_down
        ))

    print(f"Image root: {args.image_base_path}")
    print(f"Input file:   {args.json_file}")
    print(f"Output directory:   {output_dir}")
    print(f"Parallel workers:     {args.num_workers} @ {args.device}")
    print(f"System Prompt: {'DISABLED' if args.disable_system_prompt else 'ENABLED'}")
    print(f"Max Depth:   {args.max_depth}")
    print("\n" * len(chunks))


    worker_stats = []
    with multiprocessing.Pool(processes=len(chunks)) as pool:
        worker_stats = pool.starmap(worker_proc, job_args)


    merge_jsonl_to_array(input_data, shared_jsonl, out_array)


    total_correct = sum(s[0] for s in worker_stats)
    total_steps   = sum(s[1] for s in worker_stats)
    total_corr    = sum(s[2] for s in worker_stats)
    total_cases   = sum(s[3] for s in worker_stats)
    stepacc_micro = (total_correct / total_steps) if total_steps > 0 else 0.0
    avg_corr = (total_corr / total_cases) if total_cases > 0 else 0.0


    zero_corr = 0
    with open(shared_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if int(obj.get("corrections", 0)) == 0:
                zero_corr += 1
    pct_zero_corr = (zero_corr / total_cases) if total_cases > 0 else 0.0

    metrics = {
        "step_correct": int(total_correct),
        "step_total": int(total_steps),
        "StepAcc_micro": stepacc_micro,
        "total_corrections": int(total_corr),
        "avg_corrections_per_case": avg_corr,
        "pct_zero_correction": pct_zero_corr
    }
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n=== Metrics ===")
    print(f"Correct / Total Steps = {total_correct} / {total_steps}")
    print(f"StepAcc_micro         = {stepacc_micro:.4f}")
    print(f"Total Corrections     = {total_corr}")
    print(f"Avg Corrections/Case  = {avg_corr:.3f}")
    print(f"% Zero-Correction     = {pct_zero_corr:.3%}")
    print(f"Metrics written to:{metric_path}")
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()


    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")


    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)


    parser.add_argument("--image-base-path", type=str,
                        default="dataset_final")
    parser.add_argument("--json-file", type=str,
                        default="dataset_final/sft_train_test/test/json_data/task2.1_test_2k_non_uniform_sample.json")
    parser.add_argument("--hierarchy-bottom-up", type=str,
                        default="dataset_final/task2/2.1_mcq_hard/diagnosis_hierarchical_final_doc_response_1017.json")
    parser.add_argument("--hierarchy-top-down", type=str,
                        default="dataset_final/task2/2.1_mcq_hard/diagnosis_hierarchical_final_doc_response_reverse_1017.json")
    parser.add_argument("--mapping-json", type=str,
                        default="dataset_final/task2/diagnosis_mapping_final_1015.json")


    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-depth", type=int, default=12)

    # System Prompt
    parser.add_argument("--disable-system-prompt", action="store_true")

    args = parser.parse_args()
    main(args)
