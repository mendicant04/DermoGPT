import argparse
import warnings
import json
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image


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
            print(f"Warning: image file not found (ID: {item['id']}) at {image_path}. skipping.", file=sys.stderr)
            return None, None

        question = None
        ground_truth = None

        for turn in item['conversations']:
            if turn['from'] == 'human' and question is None:
                question = turn['value'].replace('<image>', '').strip()
            elif turn['from'] == 'gpt' and ground_truth is None:
                ground_truth = turn['value']

        if question is None:
            print(f"Warning: 'human' turn not found (ID: {item['id']}). skipping.", file=sys.stderr)
            return None, None
        
        if ground_truth is None:
            print(f"Warning: 'gpt' ground truth not found (ID: {item['id']}).", file=sys.stderr)
            ground_truth = ""


        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
        conversation = [{"role": "user", "content": user_content}]


        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        

        image_inputs, video_inputs = process_vision_info(conversation)
        


        if not image_inputs:
             print(f"Warning (serial mode): process_vision_info failed to load the image (ID: {item['id']}) at {image_path}. Skipping this item.", file=sys.stderr)
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

        print(f"!!! Critical error: processing ID {item.get('id', 'UNKNOWN')} failed: {e} !!!", file=sys.stderr)
        print("!!! Skipping this item !!!", file=sys.stderr)
        return None, None

def main(args):
    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    print("Loading model...")
    processor, model = load_pretrained_model(
        model_base = args.model_base, model_path = args.model_path, 
        device_map=args.device, model_name=model_name, 
        load_4bit=args.load_4bit, load_8bit=args.load_8bit,
        device=args.device, use_flash_attn=use_flash_attn
    )
    model.eval()
    print("Model loaded.")

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
        "task3_reason_one_sample_debug_1024.json"
    ]

    for json_file in json_files_to_process:
        input_file_path = os.path.join(JSON_DIR, json_file)
        

        if not args.loaded_model_name:
             print("Error: Please provide --loaded-model-name to name the output file.", file=sys.stderr)
             sys.exit(1)
        output_file_name = os.path.splitext(json_file)[0] + f"_{args.loaded_model_name}_results.jsonl"
        output_file_path = os.path.join(JSON_DIR, output_file_name)


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
            with open(input_file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"Loaded {len(input_data)} total items from {json_file}.")
        except Exception as e:
            print(f"Error: Failed to load input file {input_file_path}.skipping this file.Error: {e}", file=sys.stderr)
            continue
        

        tasks_to_run = []
        for item in input_data:
            item_id = item.get('id')
            if not item_id:
                print(f"Warning: item is missing 'id'.skipping: {str(item)[:100]}...", file=sys.stderr)
                continue
            if item_id not in processed_ids:
                tasks_to_run.append(item)
        
        print(f"Need to process {len(tasks_to_run)} new items.")


        with open(output_file_path, 'a', encoding='utf-8') as f_out:
            for item in tqdm(tasks_to_run, desc=f"Inference {json_file}"):
                

                

                ground_truth, model_response = run_inference_on_item(item, IMAGE_BASE_PATH, generation_args)
                
                if model_response is not None:

                    result = {
                        "id": item['id'],
                        "image": item['image'],
                        "ground_truth": ground_truth,
                        "model_response": model_response
                    }

                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()


        print(f"--- Finished processing: {json_file} ---")

    print("\nAll files have been processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    

    parser.add_argument("--loaded-model-name", type=str, help="model identifier used in output filenames")
    

    
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    main(args)
