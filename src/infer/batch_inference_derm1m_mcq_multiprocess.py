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


from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")


processor = None
model = None

device = "cuda"

def run_inference_on_item(item, image_base_path, generation_args):
    global processor, model, device

    try:
        image_path = os.path.join(image_base_path, item['image'])
        if not os.path.exists(image_path):
            print(f"Warning [Worker on {device}]: image file not found (ID: {item['id']}) at {image_path}. skipping.", file=sys.stderr)
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
            print(f"Warning [Worker on {device}]: 'human' turn not found (ID: {item['id']}). skipping.", file=sys.stderr)
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
             print(f"Warning [Worker on {device}]: process_vision_info failed to load the image (ID: {item['id']}) at {image_path}. skipping.", file=sys.stderr)
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
        model_response = processor.tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return ground_truth, model_response.strip()

    except Exception as e:
        print(f"!!! Critical error [Worker on {device}]: processing ID {item.get('id', 'UNKNOWN')} failed: {e} !!!", file=sys.stderr)
        return None, None

def process_task_list_worker(tasks, worker_index, device_id, output_file_path, lock, image_base_path, generation_args, cli_args):
    global processor, model, device

    device = device_id # e.g., 'cuda:0'
    


    print(f"[Worker {worker_index} on {device}]: Loading model replica...")
    disable_torch_init()
    model_name = get_model_name_from_path(cli_args.model_path)
    use_flash_attn = not cli_args.disable_flash_attention

    processor, model = load_pretrained_model(
        model_base = cli_args.model_base, model_path = cli_args.model_path, 
        device_map=device,
        model_name=model_name, 
        load_4bit=cli_args.load_4bit, load_8bit=cli_args.load_8bit,
        device=device, use_flash_attn=use_flash_attn
    )
    model.eval()
    print(f"[Worker {worker_index} on {device}]: Model loaded.starting to process {len(tasks)} items...")



    for item in tqdm(tasks, desc=f"Inference (Worker {worker_index} on {device})", position=worker_index, file=sys.stdout):
        ground_truth, model_response = run_inference_on_item(item, image_base_path, generation_args)
        
        if model_response is not None:
            result = {
                "id": item['id'],
                "image": item['image'],
                "ground_truth": ground_truth,
                "model_response": model_response
            }
            

            lock.acquire()
            try:
                with open(output_file_path, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
            finally:
                lock.release()

    print(f"[Worker {worker_index} on {device}]: Finished all tasks.")


def main(args):

    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    JSON_DIR = "dataset_final/sft_train_test/test/json_data"
    IMAGE_BASE_PATH = 'dataset_final' 
    print(f"Image base path: {IMAGE_BASE_PATH}")
    
    json_files_to_process = [
        "task2_diagnosis_mcq_hard_derm1m_edu_caseonly.json"
    ]



    devices = [args.device for _ in range(args.num_workers)]
    print(f"Launching on {args.device} with {args.num_workers} parallel worker processes...")
    manager = multiprocessing.Manager()


    for json_file in json_files_to_process:
        if not args.loaded_model_name:
             print("Error: Please provide --loaded-model-name to name the output file.", file=sys.stderr)
             sys.exit(1)
        output_file_path = os.path.join(JSON_DIR, os.path.splitext(json_file)[0] + f"_{args.loaded_model_name}_results.jsonl")

        print(f"\n--- Processing file: {json_file} ---")
        print(f"Results will be saved to: {output_file_path}")


        processed_ids = set()
        if os.path.exists(output_file_path):
            print("Existing result file found. Loading processed IDs...")
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            processed_ids.add(json.loads(line)['id'])
                        except json.JSONDecodeError:
                            print(f"Warning: skipping corrupt line in {output_file_path}: {line.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading result file {output_file_path} failed: {e}.will restart.", file=sys.stderr)
                processed_ids = set()
        print(f"Loaded {len(processed_ids)} processed IDs.")


        try:
            with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"Loaded {len(input_data)} total items from {json_file}.")
        except Exception as e:
            print(f"Error: Failed to load input file {os.path.join(JSON_DIR, json_file)}.skipping this file.Error: {e}", file=sys.stderr)
            continue
        
        tasks_to_run = []
        for item in input_data:
            item_id = item.get('id')
            if not item_id:
                continue
            if item_id not in processed_ids:
                tasks_to_run.append(item)
        
        if not tasks_to_run:
            print("No new items to process. Skipping this file.")
            continue
            
        print(f"Need to process {len(tasks_to_run)} new items.")


        lock = manager.Lock()
        

        chunk_size = math.ceil(len(tasks_to_run) / args.num_workers)
        chunks = [tasks_to_run[i:i + chunk_size] for i in range(0, len(tasks_to_run), chunk_size)]
        
        job_args = []
        for i in range(len(chunks)):
            job_args.append((
                chunks[i],
                i,
                devices[i],             # 3. device_id (e.g., 'cuda:0')
                output_file_path,
                lock,
                IMAGE_BASE_PATH,
                generation_args,
                args
            ))


        print(f"Starting {len(chunks)} worker processes... (this will load {len(chunks)} model replicas into VRAM)")

        print("\n" * args.num_workers) 
        
        with multiprocessing.Pool(processes=args.num_workers) as pool:

            pool.starmap(process_task_list_worker, job_args)

        print(f"--- Finished processing: {json_file} ---")

    print("\nAll files have been processed.")

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device ID used for all model copies (e.g., cuda:0)")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    
    parser.add_argument("--loaded-model-name", type=str, help="model identifier used in output filenames")
    

    parser.add_argument("--num-workers", type=int, default=2, help="number of parallel worker processes; each loads one model copy")
    
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
        
    main(args)
