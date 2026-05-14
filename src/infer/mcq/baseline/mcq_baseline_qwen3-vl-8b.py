import argparse
import warnings
import json
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
import multiprocessing
import math

# Project-specific imports
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# --- Global model variables (process-local) ---
processor = None
model = None
# 'device' will be set inside worker
device = "cuda"


def run_inference_on_item(item, image_base_path, generation_args):
    """
    Run inference on a single JSON entry.
    """
    global processor, model, device

    try:
        image_path = os.path.join(image_base_path, item['image'])
        if not os.path.exists(image_path):
            print(f"Warning [Worker on {device}]: Image file not found (ID: {item['id']}) at {image_path}. Skipping.", file=sys.stderr)
            return None, None

        question = None
        ground_truth = None
        for turn in item['conversations']:
            if turn['from'] == 'human' and question is None:
                question = turn['value'].replace('<image>', '').strip()
                question += 'NOTE: Respond with ONLY the letter of your choice.'
            elif turn['from'] == 'gpt' and ground_truth is None:
                ground_truth = turn['value']

        if question is None:
            print(f"Warning [Worker on {device}]: No 'human' turn found (ID: {item['id']}). Skipping.", file=sys.stderr)
            return None, None

        if ground_truth is None:
            ground_truth = ""

        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
        conversation = [{"role": "user", "content": user_content}]

        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(conversation)

        if not image_inputs:
            print(f"Warning [Worker on {device}]: process_vision_info failed to load image (ID: {item['id']}) at {image_path}. Skipping.", file=sys.stderr)
            return None, None

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generation_kwargs = dict(inputs, **generation_args)
        if processor.tokenizer.pad_token_id is None:
            generation_kwargs['pad_token_id'] = processor.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        input_token_len = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_token_len:]
        model_response = processor.tokenizer.decode(
            response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return ground_truth, model_response.strip()

    except Exception as e:
        print(f"!!! Fatal Error [Worker on {device}]: Error processing ID {item.get('id', 'UNKNOWN')}: {e} !!!", file=sys.stderr)
        return None, None


def process_task_list_worker(tasks, worker_index, device_id, output_file_path, lock, image_base_path, generation_args, cli_args):
    """
    Subprocess worker function.
    It loads its own copy of the model onto the SAME GPU and processes its assigned tasks.
    """
    global processor, model, device

    device = device_id  # e.g., 'cuda:0'

    # 1) Load model inside the subprocess
    print(f"[Worker {worker_index} on {device}]: Loading model copy...")
    disable_torch_init()
    model_name = get_model_name_from_path(cli_args.model_path)
    use_flash_attn = not cli_args.disable_flash_attention

    processor, model = load_pretrained_model(
        model_base=cli_args.model_base, model_path=cli_args.model_path,
        device_map=device,  # assign to specific GPU (e.g., 'cuda:0')
        model_name=model_name,
        load_4bit=cli_args.load_4bit, load_8bit=cli_args.load_8bit,
        device=device, use_flash_attn=use_flash_attn
    )
    model.eval()
    print(f"[Worker {worker_index} on {device}]: Model loaded. Processing {len(tasks)} items...")

    # 2) Iterate over assigned tasks
    for item in tqdm(tasks, desc=f"Inference (Worker {worker_index} on {device})", position=worker_index, file=sys.stdout):
        ground_truth, model_response = run_inference_on_item(item, image_base_path, generation_args)

        if model_response is not None:
            result = {
                "id": item['id'],
                "image": item['image'],
                "ground_truth": ground_truth,
                "model_response": model_response
            }

            # 3) Safely write to the shared output file using a lock
            lock.acquire()
            try:
                with open(output_file_path, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
            finally:
                lock.release()

    print(f"[Worker {worker_index} on {device}]: All tasks completed.")


def main(args):
    # --- Main process ---

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    if not args.image_base_path:
        print("Error: --image-base-path is required.", file=sys.stderr)
        sys.exit(1)
    if not args.json_files:
        print("Error: --json-files is required.", file=sys.stderr)
        sys.exit(1)
    if not args.output_dir:
        print("Error: --output-dir is required.", file=sys.stderr)
        sys.exit(1)

    IMAGE_BASE_PATH = args.image_base_path
    json_files_to_process = args.json_files

    print(f"Image base path set to: {IMAGE_BASE_PATH}")
    print(f"Launching {args.num_workers} parallel worker process(es) on {args.device} ...")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # All workers use the same device (potentially loading multiple copies on the same GPU)
    devices = [args.device for _ in range(args.num_workers)]
    manager = multiprocessing.Manager()

    for json_file in json_files_to_process:
        if not args.loaded_model_name:
            print("Error: Please provide --loaded-model-name to name the output files.", file=sys.stderr)
            sys.exit(1)

        # Derive output file path:
        # <output_dir>/<json_basename>_<loaded_model_name>_results.jsonl
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_file_path = os.path.join(args.output_dir, f"{base_name}_{args.loaded_model_name}_results.jsonl")

        print(f"\n--- Processing file: {json_file} ---")
        print(f"Results will be written to: {output_file_path}")

        # 1) Load processed IDs (in main process)
        processed_ids = set()
        if os.path.exists(output_file_path):
            print("Found existing results file. Loading processed IDs...")
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            processed_ids.add(json.loads(line)['id'])
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping corrupted line in {output_file_path}: {line.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading results file {output_file_path}: {e}. Starting fresh.", file=sys.stderr)
                processed_ids = set()
        print(f"Loaded {len(processed_ids)} processed ID(s).")

        # 2) Load and filter input data (in main process)
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"Loaded {len(input_data)} total entries from {json_file}.")
        except Exception as e:
            print(f"Error: Failed to load input file {json_file}. Skipping this file. Error: {e}", file=sys.stderr)
            continue

        tasks_to_run = []
        for item in input_data:
            item_id = item.get('id')
            if not item_id:
                continue
            if item_id not in processed_ids:
                tasks_to_run.append(item)

        if not tasks_to_run:
            print("No new entries to process. Skipping this file.")
            continue

        print(f"Total new entries to process: {len(tasks_to_run)}")

        # 3) Split tasks and prepare the process pool
        lock = manager.Lock()  # shared lock for all subprocesses

        chunk_size = math.ceil(len(tasks_to_run) / args.num_workers)
        chunks = [tasks_to_run[i:i + chunk_size] for i in range(0, len(tasks_to_run), chunk_size)]

        job_args = []
        for i in range(len(chunks)):
            job_args.append((
                chunks[i],              # 1. tasks for this worker
                i,                      # 2. worker_index (for tqdm)
                devices[i],             # 3. device_id (e.g., 'cuda:0')
                output_file_path,       # 4. shared output file path
                lock,                   # 5. shared lock
                IMAGE_BASE_PATH,        # 6. image base path
                generation_args,        # 7. generation args
                args                    # 8. cli args (for model loading)
            ))

        # 4) Start the process pool and run
        print(f"Starting {len(chunks)} worker process(es)... (this will load {len(chunks)} model copy/copies into VRAM)")
        print("\n" * args.num_workers)

        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pool.starmap(process_task_list_worker, job_args)

        print(f"--- Finished: {json_file} ---")

    print("\nAll files processed.")


if __name__ == "__main__":
    # Use 'spawn' for CUDA multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    # Device, image base path, and input list are now configurable
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device ID to load all model copies onto (e.g., cuda:0)")
    parser.add_argument("--image-base-path", type=str, required=True, help="Base directory for images referenced by JSON 'image' fields")
    parser.add_argument("--json-files", type=str, nargs='+', required=True, help="One or more JSON files to process (absolute or relative paths)")

    # Output directory replaces prior JSON_DIR usage for outputs
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where result .jsonl files will be saved")

    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--loaded-model-name", type=str, help="Model identifier appended to output filenames")

    # New: number of worker processes
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel worker processes (each loads a model copy)")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
