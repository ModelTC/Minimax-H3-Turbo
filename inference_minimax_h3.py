#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batch MiniMax-H3 T2VA, I2VA and FL2VA inference from a JSON job file.

MiniMax-H3 runs at 24 FPS and its video VAE accepts frame counts of the form
``17 * n + 5``. Therefore the shortest valid clip is 124 frames (about 5.17
seconds). The closest valid 16:9 540p canvas is 960x544 because both dimensions
must be divisible by 32.

``--jobs-json`` contains an ``examples`` array. Every example supplies ``task``
and the three official prompt fields. An absent/empty ``images`` list is T2VA,
one image is I2VA, and two images are FL2VA. The explicit task must agree with
the image count. Relative image paths are resolved from the JSON file.
"""

import argparse
import gc
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video
from peft import LoraConfig
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

MODEL_ID = "MiniMaxAI/MiniMax-H3"
FPS = 24
HEIGHT = 544
WIDTH = 960
NUM_FRAMES = 124
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_LORA_ALPHA = 8
LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
CORE_PROMPT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
TASK_ALIASES = {
    "t2va": "t2va",
    "i2va": "i2va",
    "i2v": "i2va",
    "fl2va": "fl2va",
    "fl2v": "fl2va",
}
TASK_BY_IMAGE_COUNT = {0: "t2va", 1: "i2va", 2: "fl2va"}


@dataclass(frozen=True)
class GenerationJob:
    task: str
    prompt: str
    image_paths: tuple[Path, ...] = ()

    @property
    def mode(self) -> str:
        return self.task


def save_result_video(result: dict, output_path: Path, fps: int = FPS) -> None:
    """Encode the first generated video and its audio into one MP4 file."""
    audio = None
    audio_sample_rate = None
    if result.get("audio") is not None:
        audio = result["audio"][0]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio = audio.detach()
        audio_sample_rate = int(result["sampling_rate"])

    encode_video(
        result["videos"][0],
        fps=fps,
        output_path=str(output_path),
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )


def _read_text(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {path}")
    return path.read_text(encoding="utf-8-sig")


def _resolve_image_path(raw_path: str, jobs_path: Path) -> Path:
    image_path = Path(raw_path).expanduser()
    if not image_path.is_absolute():
        image_path = jobs_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    return image_path


def _parse_images(
    example: Mapping, example_index: int, jobs_path: Path
) -> tuple[Path, ...]:
    if "image" in example and "images" in example:
        raise ValueError(
            f"Example {example_index} must not contain both 'image' and 'images'."
        )
    raw_images = example.get("images", example.get("image", []))
    if raw_images is None:
        raw_images = []
    elif isinstance(raw_images, str):
        raw_images = [raw_images]
    if not isinstance(raw_images, list) or not all(
        isinstance(path, str) and path.strip() for path in raw_images
    ):
        raise TypeError(
            f"Example {example_index} images must be a string or a list of paths."
        )
    if len(raw_images) > 2:
        raise ValueError(
            f"Example {example_index} has {len(raw_images)} images; at most 2 are supported."
        )
    return tuple(_resolve_image_path(path, jobs_path) for path in raw_images)


def _build_prompt(example: Mapping, example_index: int) -> str:
    sections = []
    for field in CORE_PROMPT_FIELDS:
        value = example.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Example {example_index} requires a non-empty string field {field!r}."
            )
        sections.append(f"{field}: {value.strip()}")
    return "\n\n".join(sections)


def build_jobs(jobs_json: Path) -> list[GenerationJob]:
    jobs_path = jobs_json.expanduser().resolve()
    try:
        document = json.loads(_read_text(jobs_path))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {jobs_path} at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error

    if isinstance(document, Mapping):
        examples = document.get("examples")
    else:
        examples = document
    if not isinstance(examples, list) or not examples:
        raise ValueError(
            "Jobs JSON must be a non-empty list or an object with a non-empty "
            "'examples' list."
        )

    jobs = []
    for example_index, example in enumerate(examples, start=1):
        if not isinstance(example, Mapping):
            raise TypeError(f"Example {example_index} must be a JSON object.")
        raw_task = example.get("task")
        if not isinstance(raw_task, str) or raw_task.lower() not in TASK_ALIASES:
            raise ValueError(
                f"Example {example_index} task must be one of: t2va, i2va, fl2va."
            )
        task = TASK_ALIASES[raw_task.lower()]
        image_paths = _parse_images(example, example_index, jobs_path)
        expected_task = TASK_BY_IMAGE_COUNT[len(image_paths)]
        if task != expected_task:
            raise ValueError(
                f"Example {example_index} declares task={task!r}, but "
                f"{len(image_paths)} image(s) require task={expected_task!r}."
            )
        jobs.append(
            GenerationJob(
                task=task,
                prompt=_build_prompt(example, example_index),
                image_paths=image_paths,
            )
        )
    return jobs


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resolve_lora_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    return path


def _load_lora_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        checkpoint = load_safetensors_file(path, device="cpu")
    else:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            # ``mmap`` is unavailable in older PyTorch releases.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a state-dict mapping in {path}, got {type(checkpoint).__name__}."
        )
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError(f"LoRA checkpoint contains non-string keys: {path}")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError(f"LoRA checkpoint contains non-tensor values: {path}")
    return checkpoint


def _validate_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor], path: Path
) -> int:
    lora_a = {}
    lora_b = {}
    unsupported_keys = []
    for key, tensor in state_dict.items():
        if key.endswith(LORA_A_SUFFIX):
            lora_a[key[: -len(LORA_A_SUFFIX)]] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            lora_b[key[: -len(LORA_B_SUFFIX)]] = tensor
        else:
            unsupported_keys.append(key)

    if unsupported_keys:
        preview = ", ".join(unsupported_keys[:3])
        raise ValueError(
            f"{path} is not a pure PEFT LoRA state dict; unsupported keys: {preview}"
        )
    if not lora_a:
        raise ValueError(f"No {LORA_A_SUFFIX} tensors found in {path}.")

    missing_a = sorted(lora_b.keys() - lora_a.keys())
    missing_b = sorted(lora_a.keys() - lora_b.keys())
    if missing_a or missing_b:
        raise ValueError(
            "Unpaired LoRA tensors in checkpoint: "
            f"missing A={missing_a[:3]}, missing B={missing_b[:3]}."
        )

    ranks = set()
    for module_name in lora_a:
        a_tensor = lora_a[module_name]
        b_tensor = lora_b[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {module_name} must be matrices, got "
                f"A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: "
                f"A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if not module_name.endswith(LORA_TARGET_MODULES):
            raise ValueError(
                f"Unsupported LoRA target module {module_name!r} in {path}."
            )
        ranks.add(a_tensor.shape[0])

    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks are unsupported, found {sorted(ranks)}.")
    return ranks.pop()


def load_lora_adapter(
    transformer: torch.nn.Module,
    lora_path: Path,
    lora_alpha: int,
    lora_scale: float,
    fuse_lora: bool,
) -> None:
    """Inject and load a PEFT LoRA checkpoint into the H3 transformer."""
    state_dict = _load_lora_state_dict(lora_path)
    rank = _validate_lora_state_dict(state_dict, lora_path)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=list(LORA_TARGET_MODULES),
            use_rslora=False,
        )
    )

    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(adapter_parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_parameters.keys())
    shape_mismatches = [
        (name, tuple(state_dict[name].shape), tuple(parameter.shape))
        for name, parameter in adapter_parameters.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "LoRA checkpoint is incompatible with the loaded H3 transformer: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"shape_mismatches={shape_mismatches[:3]}."
        )

    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [
        key
        for key in incompatible.missing_keys
        if ".lora_A." in key or ".lora_B." in key
    ]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "LoRA loading did not consume the expected adapter tensors: "
            f"missing={missing_lora[:3]}, "
            f"unexpected={incompatible.unexpected_keys[:3]}."
        )

    transformer.set_adapters("default", weights=lora_scale)
    if fuse_lora:
        # ``set_adapters`` has already applied ``lora_scale``. Fuse with 1.0
        # here so the runtime multiplier is not applied a second time.
        transformer.fuse_lora(
            lora_scale=1.0,
            safe_fusing=True,
            adapter_names=["default"],
        )
        transformer.unload_lora()
    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    print(
        f"Loaded LoRA: path={lora_path} tensors={tensor_count} rank={rank} "
        f"alpha={lora_alpha} scale={lora_scale} "
        f"effective_scale={lora_scale * lora_alpha / rank:.8g} "
        f"fused={fuse_lora}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--jobs-json",
        type=Path,
        required=True,
        help=(
            "UTF-8 JSON file containing mixed T2VA/I2VA/FL2VA examples and "
            "their optional image paths."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional PEFT LoRA .pt or .safetensors checkpoint.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha used during training (default: 8).",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Runtime multiplier for the loaded LoRA (default: 1.0).",
    )
    parser.add_argument(
        "--fuse-lora",
        action="store_true",
        help=(
            "Fuse the loaded LoRA into transformer weights and unload its "
            "adapter layers. Disabled by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for numbered MP4 results (default: outputs).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed (default: 42)."
    )
    parser.add_argument(
        "--inference-steps",
        "--num-inference-steps",
        dest="inference_steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help=(
            "Number of transformer evaluations per request (default: 4). The "
            "script adds the terminal sigma point required by Diffusers."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Accelerator used for inference (default: cuda).",
    )
    parser.add_argument(
        "--memory-reserve-margin",
        default="12GB",
        help="GPU memory left free by automatic CPU offload (default: 12GB).",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Keep all components on the accelerator; requires enough device memory.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help='Optional transformer attention backend, for example "_flash_3_hub".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print all requests without loading the model.",
    )
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace) -> ModularPipeline:
    dtype = torch.bfloat16
    manager = None

    if args.no_cpu_offload:
        pipe = ModularPipeline.from_pretrained(args.model_id)
        pipe.load_components(workflow="fl2va", dtype=dtype)
    else:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(
            device=args.device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
        pipe = ModularPipeline.from_pretrained(
            args.model_id, components_manager=manager
        )
        # T2VA and FL2VA share the transformer partition. Loading the FL2VA
        # component subset avoids trying to load the absent ref2va transformer.
        pipe.load_components(workflow="fl2va", dtype=dtype)

    if args.lora_path is not None:
        load_lora_adapter(
            transformer=pipe.transformer,
            lora_path=args.lora_path,
            lora_alpha=args.lora_alpha,
            lora_scale=args.lora_scale,
            fuse_lora=args.fuse_lora,
        )

    if args.no_cpu_offload:
        pipe.to(args.device)

    if args.attention_backend is not None:
        pipe.transformer.set_attention_backend(args.attention_backend)

    return pipe


def main() -> None:
    args = parse_args()
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be at least 1.")
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be at least 1.")
    if not math.isfinite(args.lora_scale) or args.lora_scale < 0:
        raise ValueError("--lora-scale must be a finite non-negative number.")
    if args.lora_path is not None:
        args.lora_path = _resolve_lora_path(args.lora_path)

    jobs = build_jobs(args.jobs_json)
    if args.lora_path is not None:
        print(
            f"lora={args.lora_path} alpha={args.lora_alpha} "
            f"scale={args.lora_scale} fuse={args.fuse_lora}",
            flush=True,
        )
    for index, job in enumerate(jobs):
        image_summary = ", ".join(map(str, job.image_paths)) or "none"
        print(
            f"request={index:04d} mode={job.mode} seed={args.seed + index} "
            f"images={image_summary} prompt={job.prompt!r}",
            flush=True,
        )
    if args.dry_run:
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args)

    # MiniMaxH3Scheduler interprets num_inference_steps as sigma grid points,
    # including terminal zero. N transformer evaluations therefore need N + 1.
    scheduler_grid_points = args.inference_steps + 1
    for index, job in enumerate(jobs):
        seed = args.seed + index
        generator = torch.Generator().manual_seed(seed)
        pipeline_kwargs = {}
        if len(job.image_paths) >= 1:
            pipeline_kwargs["image"] = _load_rgb_image(job.image_paths[0])
        if len(job.image_paths) == 2:
            pipeline_kwargs["last_image"] = _load_rgb_image(job.image_paths[1])

        with torch.inference_mode():
            result = pipe(
                prompt=job.prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=scheduler_grid_points,
                generator=generator,
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
                **pipeline_kwargs,
            )

        output_path = output_dir / f"{index:04d}_{job.mode}_seed{seed}.mp4"
        save_result_video(result, output_path, FPS)
        print(
            f"Saved {WIDTH}x{HEIGHT}, {NUM_FRAMES}-frame {job.mode} "
            f"video with muxed audio to {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
