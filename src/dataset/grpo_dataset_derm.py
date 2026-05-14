# src/dataset/grpo_dataset.py
import copy
import os
from types import SimpleNamespace
from typing import Dict, List, Any, Union

import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE as DEFAULT_SYSTEM_MESSAGE,

    SYSTEM_MESSAGE_FOR_TASK_1,
    SYSTEM_MESSAGE_FOR_TASK_3,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai


def _choose_system_message(sample: dict) -> str:
    task = (
        sample.get("ground_truth", {})
        .get("task", "")
        .strip()
        .lower()
    )
    if task == "task1":
        return SYSTEM_MESSAGE_FOR_TASK_1
    if task == "task3":
        return SYSTEM_MESSAGE_FOR_TASK_3
    raise ValueError('task not supported for grpo')


def _safe_join(folder: str, path: str) -> str:
    if os.path.isabs(path) or path.startswith("http"):
        return path
    return os.path.join(folder, path) if folder else path


class GRPODataset(Dataset):
    """Dataset for GRPO training (dermatology)."""

    def __init__(
        self,
        data_path: Union[str, List[dict]],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        padding: bool = True,
    ):
        super().__init__()
        if isinstance(data_path, str):

            if data_path.endswith(".jsonl"):
                list_data_dict = []
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            list_data_dict.append(json.loads(line))
            else:
                list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding


        # self.image_min_pixel = getattr(data_args, "image_min_pixels", 256*256)
        # self.image_max_pixel = getattr(data_args, "image_max_pixels", 1536*1536)
        # self.video_min_pixel = getattr(data_args, "video_min_pixels", 256*256)
        # self.video_max_pixel = getattr(data_args, "video_max_pixels", 1536*1536)
        # self.image_resized_w = getattr(data_args, "image_resized_width", 448)
        # self.image_resized_h = getattr(data_args, "image_resized_height", 448)
        # self.video_resized_w = getattr(data_args, "video_resized_width", 448)
        # self.video_resized_h = getattr(data_args, "video_resized_height", 448)
        # self.fps = getattr(data_args, "fps", 1)
        # self.nframes = getattr(data_args, "nframes", 8)
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes
        self.image_folder = getattr(data_args, "image_folder", "")

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False


        try:
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.do_resize = False
        except Exception:
            pass

    def __len__(self):
        return len(self.list_data_dict)

    def _load_images(self, image_files: Union[str, List[str]]):
        if isinstance(image_files, str):
            image_files = [image_files]
        images = []
        for image_file in image_files:
            image_path = _safe_join(self.image_folder, image_file)
            try:
                image_input = get_image_info(
                    image_path,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                    self.image_patch_size,
                )
                images.append(image_input)
            except Exception as e:

                print(f"[GRPO] warn: load image failed: {image_path} ({e}) -> set images=None")
                return None
        return images

    def _load_videos(self, video_files: Union[str, List[str]]):
        if isinstance(video_files, str):
            video_files = [video_files]
        videos = []
        for video_file in video_files:
            video_path = _safe_join(self.image_folder, video_file)
            try:
                video_input, _video_kwargs = get_video_info(
                    video_path,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata,
                )
                videos.append(video_input)
            except Exception as e:
                print(f"[GRPO] warn: load video failed: {video_path} ({e}) -> set videos=None")
                return None
        return videos

    def __getitem__(self, i) -> Dict[str, Any]:
        sources = self.list_data_dict[i]
        is_video = False


        images, videos = None, None
        if "image" in sources:
            images = self._load_images(sources["image"])
        elif "video" in sources:
            is_video = True
            videos = self._load_videos(sources["video"])



        conv = sources.get("conversations", [])

        messages = copy.deepcopy(llava_to_openai(conv, is_video=is_video)) if conv else []


        user_msg = None
        for m in messages:
            if m.get("role") == "user":
                user_msg = m
                break
        if user_msg is None:
            raise ValueError(f"[GRPO] sample {sources.get('id')} missing user message.")


        sys_text = _choose_system_message(sources)


        # <|im_start|>system ... <|im_end|>\n
        # <|im_start|>user   ... <|im_end|>\n

        system_block = f"{DEFAULT_IM_START_TOKEN}system\n{sys_text}{DEFAULT_IM_END_TOKEN}\n"
        user_block   = f"{DEFAULT_IM_START_TOKEN}user\n{user_msg['content']}{DEFAULT_IM_END_TOKEN}\n"
        assistant_head = f"{DEFAULT_IM_START_TOKEN}assistant\n"

        prompt = system_block + user_block + assistant_head

        data_dict = dict(
            id=sources.get("id", f"sample-{i}"),
            prompt=prompt,
            images=images,     # or None
            videos=videos,     # or None
            ground_truth=sources.get("ground_truth", {}),
        )
        return data_dict


def make_grpo_data_module_derm(model_id, processor, data_args):
    """Make dataset dict for GRPO training."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    return dict(train_dataset=grpo_dataset, eval_dataset=None)



if __name__ == "__main__":

    samples = [
        {
            "id": "grpo_task1_ISIC_0005769",
            "image": "isic/images/dermoscopic/ISIC_0005769.jpg",
            "conversations": [
                {"from": "human", "value": "Describe in detail the overall pattern and key structures seen in this dermoscopic image."}
            ],
            "ground_truth": {
                "task": "task1",
                "modality": "dermoscopic",
                "final_label": "Nevus / Mole / Melanocytic Nevus",
                "taxon_path": [
                    "Neoplasms & Proliferations",
                    "Benign Tumors, Growths & Cysts",
                    "Melanocytic Lesions (Nevi/Moles)"
                ],
                "morph": {
                    "morphological_features_Derm7pt": {
                        "Pigment Network": "absent",
                        "Blue Whitish Veil": "absent",
                        "Vascular Structures": "absent",
                        "Pigmentation": "localized irregular",
                        "Streaks": "absent",
                        "Dots and Globules": "irregular",
                        "Regression Structures": "absent"
                    }
                }
            }
        },
        {
            "id": "grpo_task3_Daffodil_SJS-TEN_xxx",
            "image": "daffodil/images/SJS-TEN/xxx.jpeg",
            "conversations": [
                {"from": "human", "value": "Based on the provided image, what is the most likely diagnosis? Please provide a detailed reasoning process before giving the final answer."}
            ],
            "ground_truth": {
                "task": "task3",
                "modality": "clinical",
                "final_label": "Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS/TEN)",
                "taxon_path": [
                    "Reactions to External Agents (Physical, Chemical, Drug-induced)",
                    "Drug Reactions"
                ],
                "morph": {
                    "morphological_features_skincon": [
                        "Black", "Brown(Hyperpigmentation)", "Patch", "Pigmented"
                    ]
                }
            }
        }
    ]


    class DummyImageProc:
        def __init__(self): self.do_resize = False
    class DummyProcessor:
        def __init__(self): self.image_processor = DummyImageProc()


    dummy_args = SimpleNamespace(
        data_path=samples,
        image_min_pixels=256*256,
        image_max_pixels=1536*1536,
        video_min_pixels=256*256,
        video_max_pixels=1536*1536,
        image_resized_width=448,
        image_resized_height=448,
        video_resized_width=448,
        video_resized_height=448,
        fps=1,
        nframes=8,
        image_folder="dataset_final",
    )

    ds = GRPODataset(
        data_path=samples,
        processor=DummyProcessor(),
        data_args=dummy_args,
        model_id="Qwen3-VL-8B",
        padding=True,
    )

    print(f"Dataset size = {len(ds)}")
    for idx in range(len(ds)):
        item = ds[idx]
        print("="*80)
        print("id:", item["id"])
        print("has_images:", item["images"] is not None)
        print("prompt_preview:\n", item["prompt"][:500].replace("\n", "\\n"))
        print("ground_truth keys:", list(item["ground_truth"].keys()))
