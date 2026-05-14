#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import sys
import warnings
from typing import Tuple, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------

# -----------------------
def build_messages(item: dict, image_base_path: str) -> Tuple[List[dict], str, str, str]:
    image_rel = item.get("image")
    if not image_rel:
        raise ValueError("missing image field")
    image_path = os.path.join(image_base_path, image_rel)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image does not exist: {image_path}")

    question = None
    ground_truth = ""
    for turn in item.get("conversations", []):
        if turn.get("from") == "human" and question is None:
            q = turn["value"].replace("<image>", "").strip()

            q += "NOTE: Respond with ONLY the letter of your choice."
            question = q
        elif turn.get("from") == "gpt" and not ground_truth:
            ground_truth = turn["value"]

    if question is None:
        raise ValueError(f"entry {item.get('id')} has no human prompt.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages, question, ground_truth, image_rel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="lingshu-medical-mllm/Lingshu-7B")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)


    parser.add_argument("--json-dir", type=str,
                        default="dataset_final/sft_train_test/test/json_data")
    parser.add_argument("--image-base-path", type=str,
                        default="dataset_final")
    parser.add_argument("--json-files", type=str,
                        default="task1.2_derm7pt_test_json_1022.json,task1.2_skincon_all_json_1022.json,task2.1_test_2k_non_uniform_sample.json")


    parser.add_argument("--loaded-model-name", type=str, required=True,
                        help="model identifier used in output filenames (e.g., Lingshu7B)")

    args = parser.parse_args()


    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation=args.attn_impl,
            device_map="auto",
        )
    except Exception:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(args.model)


    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    model.eval()
    device = model.device
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
    }


    json_files = [x.strip() for x in args.json_files.split(",") if x.strip()]
    for jf in json_files:
        in_path = os.path.join(args.json_dir, jf)
        if not os.path.exists(in_path):
            print(f"[skipping] Input file does not exist: {in_path}", file=sys.stderr)
            continue

        out_path = os.path.join(
            args.json_dir,
            os.path.splitext(jf)[0] + f"_{args.loaded_model_name}_results.jsonl"
        )
        print(f"\n===== File: {jf}")
        print(f"Output: {out_path}")


        done_ids = set()
        if os.path.exists(out_path):
            print("Found an existing result file. Loading completed item IDs...")
            with open(out_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"Completed: {len(done_ids)}")


        try:
            with open(in_path, "r", encoding="utf-8") as fin:
                data = json.load(fin)
        except Exception as e:
            print(f"[skipping] Failed to parse {in_path}: {e}", file=sys.stderr)
            continue


        tasks = [it for it in data if it.get("id") and it["id"] not in done_ids]
        if not tasks:
            print("No new items to process.")
            continue

        print(f"Pending: {len(tasks)} (sequential inference, batch_size=1)")


        with open(out_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(tasks), desc="Inference", dynamic_ncols=True)
            for item in tasks:
                try:
                    messages, _q_for_model, gt, image_rel = build_messages(item, args.image_base_path)


                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


                    image_inputs, video_inputs = process_vision_info(messages)

                    if not image_inputs:
                        print(f"[skipping] ID={item.get('id')}: process_vision_info failed to load the image", file=sys.stderr)
                        pbar.update(1)
                        continue


                    proc_kwargs = dict(
                        text=[prompt],
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    if video_inputs and len(video_inputs) > 0:
                        proc_kwargs["videos"] = video_inputs

                    inputs = processor(**proc_kwargs).to(device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_args)


                    input_len = inputs["input_ids"].shape[1]
                    new_tokens = outputs[0][input_len:]
                    model_text = processor.decode(
                        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    rec = {
                        "id": item["id"],
                        "image": image_rel,
                        "ground_truth": gt,
                        "model_response": model_text
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[skipping] ID={item.get('id')}: CUDA OOM", file=sys.stderr)
                except Exception as e:
                    print(f"[skipping] ID={item.get('id')}: {e}", file=sys.stderr)
                finally:
                    pbar.update(1)
            pbar.close()

    print("\nAll evaluations completed.")


if __name__ == "__main__":
    main()
