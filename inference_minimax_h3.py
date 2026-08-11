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
"""Batch MiniMax-H3 T2VA, I2VA, FL2VA and Ref2VA inference from JSON.

MiniMax-H3 runs at 24 FPS and its video VAE accepts frame counts of the form
``17 * n + 5``. Therefore the shortest valid clip is 124 frames (about 5.17
seconds). The closest valid 16:9 540p canvas is 960x544 because both dimensions
must be divisible by 32.

``--jobs-json`` contains an ``examples`` array. Every example supplies ``task``
and the three official prompt fields. An absent/empty ``images`` list is T2VA,
one image is I2VA, and two images are FL2VA. Ref2VA instead supplies an ordered
``references`` list of image, video, and/or audio media. Relative media paths
are resolved from the JSON file.
"""

import argparse
import gc
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils.export_utils import encode_video
from peft import LoraConfig
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file

MODEL_ID = "/mnt/aigc/fanxiangyu/repos/video_gen/Bagel/Minimax-H3"
FPS = 24
HEIGHT = 544
WIDTH = 960
NUM_FRAMES = 124
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
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
    "ref2va": "ref2va",
    "ref2v": "ref2va",
}
TASK_BY_IMAGE_COUNT = {0: "t2va", 1: "i2va", 2: "fl2va"}
REFERENCE_TYPES = ("image", "video", "audio")
REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
MAX_REFERENCES = 12


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    path: Path


@dataclass(frozen=True)
class GenerationJob:
    task: str
    prompt: str
    image_paths: tuple[Path, ...] = ()
    reference_specs: tuple[ReferenceSpec, ...] = ()

    @property
    def mode(self) -> str:
        return self.task

    @property
    def workflow(self) -> str:
        return "ref2va" if self.task == "ref2va" else "fl2va"


@dataclass(frozen=True)
class AssignedJob:
    global_index: int
    job: GenerationJob
    is_padding: bool = False


@dataclass(frozen=True)
class DistributedContext:
    fsdp2: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.local_rank)


def distributed_context_from_env(enable_fsdp2: bool) -> DistributedContext:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not enable_fsdp2:
        if env_world_size > 1:
            raise RuntimeError(
                "Detected a multi-process launch without --fsdp2. Add --fsdp2 "
                "to enable model sharding and rank-local JSON assignment."
            )
        return DistributedContext()

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "--fsdp2 must be launched with torchrun; missing environment "
            f"variables: {', '.join(missing)}."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            "Invalid distributed environment: "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}."
        )
    return DistributedContext(
        fsdp2=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def assign_jobs_to_rank(
    jobs: list[GenerationJob], context: DistributedContext
) -> tuple[list[AssignedJob], int]:
    """Shard jobs like DistributedSampler(drop_last=False), padding if needed."""
    if not jobs:
        raise ValueError("Cannot distribute an empty job list.")
    if context.world_size == 1:
        return [AssignedJob(index, job) for index, job in enumerate(jobs)], 0

    jobs_per_rank = math.ceil(len(jobs) / context.world_size)
    total_size = jobs_per_rank * context.world_size
    padding_size = total_size - len(jobs)
    indexed_jobs = [AssignedJob(index, job) for index, job in enumerate(jobs)]
    if padding_size:
        repetitions = math.ceil(padding_size / len(indexed_jobs))
        padding_source = (indexed_jobs * repetitions)[:padding_size]
        indexed_jobs.extend(
            AssignedJob(item.global_index, item.job, is_padding=True)
            for item in padding_source
        )
    return indexed_jobs[context.rank:total_size:context.world_size], padding_size


def initialize_fsdp2(context: DistributedContext):
    if not context.fsdp2:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP2 requires CUDA, but CUDA is not available.")
    if context.local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={context.local_rank} is out of range for "
            f"{torch.cuda.device_count()} visible CUDA devices."
        )

    torch.cuda.set_device(context.local_rank)
    dist.init_process_group(backend="nccl", device_id=context.device)
    if dist.get_rank() != context.rank or dist.get_world_size() != context.world_size:
        raise RuntimeError(
            "torch.distributed does not match the torchrun environment: "
            f"process_group=({dist.get_rank()}, {dist.get_world_size()}), "
            f"environment=({context.rank}, {context.world_size})."
        )

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "This PyTorch build does not provide FSDP2. Install PyTorch 2.4 or newer."
        ) from error
    return init_device_mesh(
        "cuda", (context.world_size,), mesh_dim_names=("fsdp",)
    )


def apply_fsdp2_to_transformer(transformer: torch.nn.Module, mesh) -> None:
    """Shard H3 bottom-up while keeping every FSDP group dtype-uniform."""
    from torch.distributed.fsdp import fully_shard

    token_refiner = getattr(transformer, "token_refiner", None)
    refiner_blocks = getattr(token_refiner, "refiner_blocks", None)
    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    if refiner_blocks is None or transformer_blocks is None:
        raise TypeError(
            "Loaded transformer does not expose the expected MiniMax-H3 block layout."
        )

    for block in refiner_blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(token_refiner, mesh=mesh, reshard_after_forward=True)
    for block in transformer_blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)

    # The released H3 checkpoint deliberately keeps these root modules in
    # different dtypes: projections/time MLP/output heads are FP32, while the
    # context projection and output norm are BF16. FSDP2 requires one original
    # dtype per parameter group, so shard each module separately before the
    # parameter-free root group is installed.
    root_modules = (
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "time_embedder",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    )
    for module_name in root_modules:
        module = getattr(transformer, module_name, None)
        if module is None:
            raise TypeError(f"Loaded H3 transformer is missing {module_name!r}.")
        fully_shard(module, mesh=mesh, reshard_after_forward=True)
    fully_shard(transformer, mesh=mesh, reshard_after_forward=True)


def apply_fsdp2_to_text_encoder_model(
    text_encoder_model: torch.nn.Module, mesh
) -> None:
    """Shard Qwen3-VL language layers and its directly-called model root."""
    from torch.distributed.fsdp import fully_shard

    language_model = getattr(text_encoder_model, "language_model", None)
    language_layers = getattr(language_model, "layers", None)
    if language_model is None or language_layers is None:
        raise TypeError(
            "Loaded text_encoder.model does not expose the expected Qwen3-VL layout."
        )

    for layer in language_layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)

    # Do not make language_model another FSDP root. Qwen3VLModel.forward reads
    # language_model.embed_tokens through get_input_embeddings() *before* it
    # calls language_model.forward. A language_model root would therefore leave
    # the embedding weight as a DTensor until too late, while input_ids remains
    # a regular Tensor. Keep embed_tokens and the final norm in the directly
    # called text_encoder.model root; only decoder layers are nested roots.
    # Diffusers calls text_encoder.model(...) directly. Shard that exact module
    # so its root hooks run on every rank, including text-only ranks that skip
    # the conditioner's visual branch.
    fully_shard(text_encoder_model, mesh=mesh, reshard_after_forward=True)

    # PyTorch keeps the outermost FSDP root unsharded after forward as a
    # training optimization. H3 only needs this conditioner once per request,
    # so explicitly release the full root (mostly vision) weights before the
    # denoising loop starts. This hook is registered after FSDP's own hooks.
    def reshard_conditioner_root(module, _inputs, output):
        module.reshard()
        return output

    text_encoder_model.register_forward_hook(reshard_conditioner_root)


def move_unsharded_components_to_device(
    pipe: ModularPipeline,
    active_transformer: torch.nn.Module,
    device: torch.device,
) -> None:
    """Move replicated components without touching either FSDP2 model."""
    fsdp_components = {id(active_transformer), id(pipe.text_encoder)}
    for component in pipe.components.values():
        if (
            isinstance(component, torch.nn.Module)
            and id(component) not in fsdp_components
        ):
            component.to(device)


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


def _resolve_media_path(raw_path: str, jobs_path: Path, media_kind: str) -> Path:
    media_path = Path(raw_path).expanduser()
    if not media_path.is_absolute():
        media_path = jobs_path.parent / media_path
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise FileNotFoundError(
            f"{media_kind.capitalize()} path does not exist: {media_path}"
        )
    return media_path


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
    return tuple(_resolve_media_path(path, jobs_path, "image") for path in raw_images)


def _parse_references(
    example: Mapping, example_index: int, jobs_path: Path
) -> tuple[ReferenceSpec, ...]:
    raw_references = example.get("references")
    if raw_references is None:
        return ()
    if not isinstance(raw_references, list):
        raise TypeError(f"Example {example_index} references must be a list.")

    references = []
    for reference_index, raw_reference in enumerate(raw_references, start=1):
        if not isinstance(raw_reference, Mapping):
            raise TypeError(
                f"Example {example_index} reference {reference_index} must be an object."
            )
        raw_kind = raw_reference.get("type")
        if not isinstance(raw_kind, str) or raw_kind.lower() not in REFERENCE_TYPES:
            raise ValueError(
                f"Example {example_index} reference {reference_index} type must be "
                f"one of: {', '.join(REFERENCE_TYPES)}."
            )
        raw_path = raw_reference.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"Example {example_index} reference {reference_index} requires "
                "a non-empty path."
            )
        kind = raw_kind.lower()
        references.append(
            ReferenceSpec(
                kind=kind,
                path=_resolve_media_path(raw_path, jobs_path, f"reference {kind}"),
            )
        )

    if len(references) > MAX_REFERENCES:
        raise ValueError(
            f"Example {example_index} has {len(references)} references; "
            f"at most {MAX_REFERENCES} are supported."
        )
    for kind, limit in REFERENCE_LIMITS.items():
        count = sum(reference.kind == kind for reference in references)
        if count > limit:
            raise ValueError(
                f"Example {example_index} has {count} {kind} references; "
                f"at most {limit} are supported."
            )
    if references and all(reference.kind == "audio" for reference in references):
        raise ValueError(
            f"Example {example_index} cannot use only audio references; add at "
            "least one image or video reference."
        )
    return tuple(references)


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
                f"Example {example_index} task must be one of: "
                "t2va, i2va, fl2va, ref2va."
            )
        task = TASK_ALIASES[raw_task.lower()]
        if task == "ref2va":
            if "image" in example or "images" in example:
                raise ValueError(
                    f"Example {example_index} is ref2va and must use references, "
                    "not image/images."
                )
            image_paths = ()
            reference_specs = _parse_references(example, example_index, jobs_path)
            if not reference_specs:
                raise ValueError(
                    f"Example {example_index} is ref2va and requires at least "
                    "one reference."
                )
        else:
            if example.get("references") not in (None, []):
                raise ValueError(
                    f"Example {example_index} declares task={task!r}; references "
                    "are only valid for ref2va."
                )
            image_paths = _parse_images(example, example_index, jobs_path)
            reference_specs = ()
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
                reference_specs=reference_specs,
            )
        )
    return jobs


def resolve_jobs_workflow(jobs: list[GenerationJob]) -> str:
    workflows = {job.workflow for job in jobs}
    if len(workflows) != 1:
        raise ValueError(
            "Ref2VA and T2VA/I2VA/FL2VA cannot be mixed in one invocation. "
            "Split them into separate JSON files so every FSDP2 rank executes "
            "the same transformer partition."
        )
    return workflows.pop()


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_references(
    reference_specs: tuple[ReferenceSpec, ...],
) -> list[MiniMaxH3ImageReference | MiniMaxH3VideoReference | MiniMaxH3AudioReference]:
    references = []
    for reference in reference_specs:
        if reference.kind == "image":
            loaded = MiniMaxH3ImageReference.from_file(reference.path)
        elif reference.kind == "video":
            loaded = MiniMaxH3VideoReference.from_file(reference.path)
        else:
            loaded = MiniMaxH3AudioReference.from_file(reference.path)
        references.append(loaded)
    return references


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
            "UTF-8 JSON containing T2VA/I2VA/FL2VA examples or Ref2VA "
            "examples. Ref2VA cannot be mixed with the base workflows."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=(
            "Full MiniMax-H3 model repository containing both transformer and "
            "transformer_ref partitions."
        ),
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help=(
            "Optional PEFT LoRA checkpoint applied to the active transformer "
            "partition."
        ),
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
            "Fuse the loaded LoRA into active transformer weights and unload its "
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
        "--video-shift",
        "--video_shift",
        dest="video_shift",
        type=float,
        default=DEFAULT_VIDEO_SHIFT,
        help="Sigma schedule shift for video latents (default: 12.0).",
    )
    parser.add_argument(
        "--audio-shift",
        "--audio_shift",
        dest="audio_shift",
        type=float,
        default=DEFAULT_AUDIO_SHIFT,
        help="Sigma schedule shift for audio latents (default: 3.0).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Accelerator used for single-process inference (default: cuda). "
            "FSDP2 always uses cuda:LOCAL_RANK."
        ),
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
        "--fsdp2",
        action="store_true",
        help=(
            "Shard the H3 text encoder and active transformer across torchrun ranks "
            "with PyTorch FSDP2. This disables CPU offload and "
            "data-parallelizes the JSON jobs."
        ),
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


def get_active_transformer(
    pipe: ModularPipeline, workflow: str
) -> tuple[str, torch.nn.Module]:
    component_name = "transformer_ref" if workflow == "ref2va" else "transformer"
    transformer = getattr(pipe, component_name, None)
    if transformer is None:
        raise RuntimeError(
            f"Workflow {workflow!r} requires {component_name!r}, but it was not loaded."
        )
    return component_name, transformer


def load_pipeline(
    args: argparse.Namespace,
    context: DistributedContext,
    workflow: str,
    fsdp_mesh=None,
) -> ModularPipeline:
    dtype = torch.bfloat16

    if args.fsdp2 or args.no_cpu_offload:
        pipe = ModularPipeline.from_pretrained(args.model_id)
    else:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(
            device=args.device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
        pipe = ModularPipeline.from_pretrained(
            args.model_id, components_manager=manager
        )

    # T2VA/I2VA/FL2VA use transformer; Ref2VA uses transformer_ref. Loading one
    # workflow keeps the inactive ~66 GB partition out of host and device memory.
    # MiniMax-H3's modular index records MiniMaxAI/MiniMax-H3 as the source of
    # every component even when the index itself was downloaded locally. An
    # explicit override is therefore required to load transformer/text encoder
    # weights from a local --model-id instead of silently falling back to that
    # recorded Hub ID.
    pipe.load_components(
        workflow=workflow,
        dtype=dtype,
        pretrained_model_name_or_path=args.model_id,
    )
    transformer_name, active_transformer = get_active_transformer(pipe, workflow)
    required_components = (
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "scheduler",
        "audio_scheduler",
        transformer_name,
        "audio_vae",
    )
    missing_components = [
        name for name in required_components if getattr(pipe, name, None) is None
    ]
    if missing_components:
        raise RuntimeError(
            f"Failed to load required components from {args.model_id!r}: "
            f"{', '.join(missing_components)}. Check the preceding Diffusers "
            "component-loading warning for the original exception."
        )

    pipe.scheduler.set_shift(args.video_shift)
    pipe.audio_scheduler.set_shift(args.audio_shift)
    print(
        f"scheduler shifts: video={pipe.scheduler.shift:g} "
        f"audio={pipe.audio_scheduler.shift:g}",
        flush=True,
    )

    if args.lora_path is not None:
        load_lora_adapter(
            transformer=active_transformer,
            lora_path=args.lora_path,
            lora_alpha=args.lora_alpha,
            lora_scale=args.lora_scale,
            fuse_lora=args.fuse_lora,
        )

    if args.attention_backend is not None:
        active_transformer.set_attention_backend(args.attention_backend)

    active_transformer.requires_grad_(False)
    active_transformer.eval()
    if args.fsdp2:
        if fsdp_mesh is None:
            raise RuntimeError("FSDP2 was requested but its device mesh is missing.")
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
        apply_fsdp2_to_text_encoder_model(pipe.text_encoder.model, fsdp_mesh)
        apply_fsdp2_to_transformer(active_transformer, fsdp_mesh)
        move_unsharded_components_to_device(
            pipe, active_transformer, context.device
        )
        print(
            f"[rank {context.rank}/{context.world_size}] FSDP2 enabled for "
            f"text encoder and {transformer_name} on {context.device}; "
            "CPU offload disabled",
            flush=True,
        )
    elif args.no_cpu_offload:
        pipe.to(args.device)

    print(
        f"workflow={workflow} active_transformer={transformer_name} ",
        f"model={args.model_id}",
        flush=True,
    )
    return pipe


def main() -> None:
    args = parse_args()
    context = distributed_context_from_env(args.fsdp2)
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be at least 1.")
    if not math.isfinite(args.video_shift) or args.video_shift <= 0:
        raise ValueError("--video-shift must be a finite positive number.")
    if not math.isfinite(args.audio_shift) or args.audio_shift <= 0:
        raise ValueError("--audio-shift must be a finite positive number.")
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be at least 1.")
    if not math.isfinite(args.lora_scale) or args.lora_scale < 0:
        raise ValueError("--lora-scale must be a finite non-negative number.")
    if args.lora_path is not None:
        args.lora_path = _resolve_lora_path(args.lora_path)

    jobs = build_jobs(args.jobs_json)
    workflow = resolve_jobs_workflow(jobs)
    assigned_jobs, padding_size = assign_jobs_to_rank(jobs, context)
    if context.rank == 0 and context.world_size > 1:
        print(
            f"Distributed jobs: original={len(jobs)} padding={padding_size} "
            f"total={len(jobs) + padding_size} world_size={context.world_size} "
            f"jobs_per_rank={len(assigned_jobs)}",
            flush=True,
        )
    if args.lora_path is not None:
        print(
            f"[rank {context.rank}] lora={args.lora_path} alpha={args.lora_alpha} "
            f"scale={args.lora_scale} fuse={args.fuse_lora}",
            flush=True,
        )
    for assignment in assigned_jobs:
        index = assignment.global_index
        job = assignment.job
        image_summary = ", ".join(map(str, job.image_paths)) or "none"
        reference_summary = (
            ", ".join(
                f"{reference.kind}:{reference.path}"
                for reference in job.reference_specs
            )
            or "none"
        )
        print(
            f"rank={context.rank:04d} request={index:04d} "
            f"padding={assignment.is_padding} mode={job.mode} "
            f"seed={args.seed + index} "
            f"workflow={workflow} images={image_summary} "
            f"references={reference_summary} prompt={job.prompt!r}",
            flush=True,
        )
    if args.dry_run:
        return

    if not args.fsdp2 and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fsdp_mesh = initialize_fsdp2(context)
        pipe = load_pipeline(args, context, workflow, fsdp_mesh)

        # MiniMaxH3Scheduler interprets num_inference_steps as sigma grid points,
        # including terminal zero. N transformer evaluations therefore need N + 1.
        scheduler_grid_points = args.inference_steps + 1
        rank_suffix = f"_rank{context.rank:04d}" if args.fsdp2 else ""
        for assignment in assigned_jobs:
            index = assignment.global_index
            job = assignment.job
            seed = args.seed + index
            generator = torch.Generator().manual_seed(seed)
            pipeline_kwargs = {}
            if job.task == "ref2va":
                pipeline_kwargs["references"] = _load_references(
                    job.reference_specs
                )
            elif len(job.image_paths) >= 1:
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

            output_path = output_dir / (
                f"{index:04d}_{job.mode}_seed{seed}{rank_suffix}.mp4"
            )
            save_result_video(result, output_path, FPS)
            print(
                f"[rank {context.rank}] Saved {WIDTH}x{HEIGHT}, "
                f"{NUM_FRAMES}-frame {job.mode} video with muxed audio to "
                f"{output_path}",
                flush=True,
            )
            del result, pipeline_kwargs
            if job.task == "ref2va":
                gc.collect()
        if context.fsdp2:
            dist.barrier()
    finally:
        if context.fsdp2 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
