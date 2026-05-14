# coding: utf-8

import argparse, warnings, json, os, sys, torch, math, multiprocessing, re
from tqdm import tqdm
from PIL import Image  # noqa: F401
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# =========================
# System Prompt (Task 3.1)
# =========================

SYS_PROMPT_T31 = (
    "You are a dermatology VQA assistant.\n"
    "Output EXACTLY TWO blocks in this order and nothing else:\n"
    "1) <reasoning> Provide a concise, step-by-step, image-grounded chain-of-thought reasoning process"
    "No probabilities, disclaimers, or instructions.</reasoning>\n"
    "2) <final_diagnosis>ONE most likely diagnosis (free-text clinical term). No extra words.</final_diagnosis>\n"
    "\n"
    "Strict rules:\n"
    "- Do NOT echo the question; do NOT add markdown, code fences, or labels such as 'Answer:'.\n"
    "- If uncertain, still pick the single most likely diagnosis based on visible cues.\n"
    "- Do NOT include patient management, tests, or treatments.\n"
)

# =========================

# =========================

RE_BLOCKS = re.compile(
    r"<reasoning>.*?</reasoning>\s*<final_diagnosis>.*?</final_diagnosis>",
    re.S
)

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
        raise ValueError(f"{file_path} top-level JSON value is not an array. Use a .jsonl suffix for JSONL input.")
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

processor = None
model = None
device = "cuda"

def build_conversation_for_model(item, image_abs_path, sys_prompt_text=None):

    human_turn = None
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            human_turn = turn
            break
    if human_turn is None:
        raise ValueError(f"sample {item.get('id')} is missing a human turn")
    raw = human_turn.get("value", "")
    question_text = raw.replace("<image>", "", 1).lstrip("\n")

    conversation = []
    if sys_prompt_text:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt_text}]
        })
    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_abs_path},
            {"type": "text", "text": question_text}
        ]
    })
    return conversation

def warn_if_format_off(model_text):
    if not RE_BLOCKS.search(model_text or ""):
        print("[Warning] Strict two-block output (<reasoning>/<final_diagnosis>) was not detected or is out of order.", file=sys.stderr)

def run_inference_on_item(item, image_base_path, generation_args,
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

    sys_prompt = SYS_PROMPT_T31 if use_system_prompt else None

    try:
        conversation = build_conversation_for_model(item, image_abs, sys_prompt_text=sys_prompt)
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
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
            warn_if_format_off(model_text)


        out_item = {"id": item.get("id"), "image": item.get("image"), "conversations": []}

        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                out_item["conversations"].append({"from": "human", "value": turn.get("value", "")})
                break
        if not out_item["conversations"]:
            out_item["conversations"].append({"from": "human", "value": ""})
        # gpt
        out_item["conversations"].append({"from": "gpt", "value": model_text})
        return out_item

    except Exception as e:
        print(f"[{device}] Critical error: processing ID {item.get('id')} failed: {e}", file=sys.stderr)
        return None

# =========================

# =========================

def worker_proc(tasks, worker_idx, device_id, image_base_path, generation_args, cli_args,
                shared_jsonl_path, lock, processed_ids):
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
    print(f"System Prompt: {'DISABLED' if args.disable_system_prompt else 'Task3.1(two blocks)'}")
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
                shared_jsonl, lock, processed_ids
            ))

        print(f"\n--- Processing {json_file}({len(input_data)} items)-> {shared_jsonl}")
        print("\n" * len(chunks))
        with multiprocessing.Pool(processes=len(chunks)) as pool:
            pool.starmap(worker_proc, job_args)


        merge_jsonl_to_array(input_data, shared_jsonl, out_array)
        print(f"--- Done: {json_file}\n")

    print("All files have been processed.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device used to load the model, e.g. cuda:0")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")


    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)


    parser.add_argument("--image-base-path", type=str, default="dataset_final", help="dataset root used to resolve relative image paths")
    parser.add_argument("--json-dir", type=str, default="dataset_final/sft_train_test/test/task1_3_benchmark_json", help="directory containing input and output JSON files")
    parser.add_argument("--json-files", type=str, default="task3_1_short.jsonl", help="comma-separated filenames; leave empty to process all .json/.jsonl files in the directory")


    parser.add_argument("--num-workers", type=int, default=1, help="number of parallel processes; each loads one model copy")
    parser.add_argument("--output-suffix", type=str, default="", help="optional suffix for output filenames")
    parser.add_argument("--output-dir", type=str, default="", help="directory for JSONL/JSON outputs; defaults to --json-dir")


    parser.add_argument("--disable-system-prompt", action="store_true", help="disable the system prompt, useful for fine-tuned models")
    parser.add_argument("--warn-format", action="store_true", help="log warnings only when <reasoning>/<final_diagnosis> blocks are missing")

    args = parser.parse_args()
    main(args)
