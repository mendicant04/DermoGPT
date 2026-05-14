import argparse
import warnings
import json
import os
import sys
import math
import multiprocessing
import time
import base64
import mimetypes
import threading
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm
from PIL import Image  # noqa: F401

warnings.filterwarnings("ignore")

# --- Global (for logging only) ---
device = "api"


ERROR_THRESHOLD = 100
PAUSE_DURATION_SECONDS = 20 * 60
IO_SEMA = threading.Semaphore(4)


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
    with IO_SEMA:
        with Image.open(image_path) as im:
            # Clamp very large images to reduce payload size and API-side parsing pressure.
            max_side = 2048
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")

            buf = BytesIO()
            im.save(buf, format="JPEG", quality=90, optimize=True)
            raw = buf.getvalue()
    mime = "image/jpeg"
    # Safety fallback, should not hit after forced JPEG conversion.
    if not raw:
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime:
            mime = "application/octet-stream"
        raw = image_path.read_bytes()
    with IO_SEMA:
        b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_chat(provider: ApiProvider, messages: list,
              max_tokens: int = 128,
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
        except requests.HTTPError as e:
            provider.record_failure()
            status = getattr(e.response, "status_code", "NA")
            body = ""
            try:
                body = (e.response.text or "").strip()
            except Exception:
                body = ""
            if len(body) > 600:
                body = body[:600] + "...(truncated)"
            tqdm.write(
                f"[RETRY {a+1}/{retry}] {provider.name} -> HTTP {status} {e} | body={body}"
            )
            time.sleep(1.0 * (a + 1))
        except Exception as e:
            provider.record_failure()
            tqdm.write(f"[RETRY {a+1}/{retry}] {provider.name} -> {e}")
            time.sleep(1.0 * (a + 1))
    raise RuntimeError("API call failed after retries")



def run_inference_on_item(item, image_base_path, generation_args, provider: ApiProvider):
    global device

    try:
        image_rel = item.get("image")
        if not image_rel:
            print(
                f"Warning [Worker on {device}]: No 'image' field (ID: {item.get('id')}). Skipping.",
                file=sys.stderr,
            )
            return None, None

        image_path = os.path.join(image_base_path, image_rel)
        if not os.path.exists(image_path):
            print(
                f"Warning [Worker on {device}]: Image file not found (ID: {item.get('id')}) at {image_path}. Skipping.",
                file=sys.stderr,
            )
            return None, None

        question = None
        ground_truth = None
        for turn in item.get("conversations", []):
            if turn.get("from") == "human" and question is None:

                question = turn.get("value", "").replace("<image>", "").strip()
                question += "NOTE: Respond with ONLY the letter of your choice."
            elif turn.get("from") == "gpt" and ground_truth is None:
                ground_truth = turn.get("value")

        if question is None:
            print(
                f"Warning [Worker on {device}]: No 'human' turn found (ID: {item.get('id')}). Skipping.",
                file=sys.stderr,
            )
            return None, None

        if ground_truth is None:
            ground_truth = ""


        img_url = b64_image_url(Path(image_path))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        max_new_tokens = int(generation_args.get("max_new_tokens", 128))
        temperature = float(generation_args.get("temperature", 0.0))

        model_response = call_chat(
            provider,
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        return ground_truth, model_response.strip()

    except Exception as e:
        print(
            f"!!! Fatal Error [Worker on {device}]: Error processing ID {item.get('id', 'UNKNOWN')}: {e} !!!",
            file=sys.stderr,
        )
        return None, None



def process_task_list_worker(tasks, worker_index, device_id, output_file_path, lock,
                             image_base_path, generation_args, cli_args):
    global device
    device = f"api-worker-{worker_index}"

    print(f"[Worker {worker_index} on {device}]: Initializing API provider...")
    provider = ApiProvider(
        name=f"Infer-{worker_index}",
        base_url=cli_args.base_url,
        api_key=cli_args.api_key,
        model=cli_args.model,
        max_workers=1,
        timeout=(10, 180),
    )
    # provider.ping()
    print(
        f"[Worker {worker_index} on {device}]: Provider ready. Processing {len(tasks)} items...",
        flush=True,
    )

    for item in tqdm(
        tasks,
        desc=f"Inference (Worker {worker_index} on {device})",
        position=worker_index,
        file=sys.stdout,
    ):
        ground_truth, model_response = run_inference_on_item(
            item, image_base_path, generation_args, provider
        )

        if model_response is not None:
            result = {
                "id": item["id"],
                "image": item["image"],
                "ground_truth": ground_truth,
                "model_response": model_response,
            }


            lock.acquire()
            try:
                with open(output_file_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
            finally:
                lock.release()

    print(f"[Worker {worker_index} on {device}]: All tasks completed.")



def main(args):
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
    print(f"API base: {args.base_url}")
    print(f"API model: {args.model}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)


    devices = [args.device for _ in range(args.num_workers)]
    manager = multiprocessing.Manager()

    for json_file in json_files_to_process:
        if not args.loaded_model_name:
            print(
                "Error: Please provide --loaded-model-name to name the output files.",
                file=sys.stderr,
            )
            sys.exit(1)

        # <output_dir>/<json_basename>_<loaded_model_name>_results.jsonl
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_file_path = os.path.join(
            args.output_dir, f"{base_name}_{args.loaded_model_name}_results.jsonl"
        )

        print(f"\n--- Processing file: {json_file} ---")
        print(f"Results will be written to: {output_file_path}")


        processed_ids = set()
        if os.path.exists(output_file_path):
            print("Found existing results file. Loading processed IDs...")
            try:
                with open(output_file_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        try:
                            processed_ids.add(json.loads(line)["id"])
                        except json.JSONDecodeError:
                            print(
                                f"Warning: Skipping corrupted line in {output_file_path}: {line.strip()}",
                                file=sys.stderr,
                            )
            except Exception as e:
                print(
                    f"Error reading results file {output_file_path}: {e}. Starting fresh.",
                    file=sys.stderr,
                )
                processed_ids = set()
        print(f"Loaded {len(processed_ids)} processed ID(s).")


        try:
            with open(json_file, "r", encoding="utf-8") as f:
                input_data = json.load(f)
            print(f"Loaded {len(input_data)} total entries from {json_file}.")
        except Exception as e:
            print(
                f"Error: Failed to load input file {json_file}. Skipping this file. Error: {e}",
                file=sys.stderr,
            )
            continue


        tasks_to_run = []
        for item in input_data:
            item_id = item.get("id")
            if not item_id:
                continue
            if item_id not in processed_ids:
                tasks_to_run.append(item)

        if not tasks_to_run:
            print("No new entries to process. Skipping this file.")
            continue

        print(f"Total new entries to process: {len(tasks_to_run)}")


        lock = manager.Lock()  # shared lock for all subprocesses

        chunk_size = math.ceil(len(tasks_to_run) / args.num_workers)
        chunks = [
            tasks_to_run[i : i + chunk_size]
            for i in range(0, len(tasks_to_run), chunk_size)
        ]

        job_args = []
        for i in range(len(chunks)):
            job_args.append(
                (
                    chunks[i],          # 1. tasks
                    i,                  # 2. worker_index
                    devices[i],
                    output_file_path,   # 4. shared output file path
                    lock,               # 5. shared lock
                    IMAGE_BASE_PATH,    # 6. image base path
                    generation_args,    # 7. generation args
                    args,
                )
            )


        print(
            f"Starting {len(chunks)} worker process(es)... (this will create {len(chunks)} API clients)"
        )
        print("\n" * args.num_workers)

        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pool.starmap(process_task_list_worker, job_args)

        print(f"--- Finished: {json_file} ---")

    print("\nAll files processed.")


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")


    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device ID for compatibility; used only in logs",
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        required=True,
        help="Base directory for images referenced by JSON 'image' fields",
    )
    parser.add_argument(
        "--json-files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSON files to process (absolute or relative paths)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where result .jsonl files will be saved",
    )

    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)

    parser.add_argument(
        "--loaded-model-name",
        type=str,
        help="Model identifier appended to output filenames",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel worker processes (each creates an API client)",
    )

    parser.add_argument("--debug", action="store_true")


    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("BASE_URL", ""),
        help="API gateway base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key; pass it on the command line or through the environment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("API_MODEL", ""),
        help="closed-source model name; gateway model identifier",
    )

    args = parser.parse_args()
    main(args)
