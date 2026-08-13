"""MiniMax-H3 Ref2VA pipeline with configurable reference-image resizing.

This module targets the Diffusers installation used by zoe-diffusion. It keeps
the stock MiniMax-H3 modular pipeline and replaces only the Ref2VA setup branch
inside ``MiniMaxH3Blocks``.
"""

import math

import numpy as np
import torch
from PIL import Image

from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep, logger
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoBeforeEncodeStep,
    MiniMaxH3Blocks,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MiniMaxH3ModularPipeline,
    align_num_frames,
    resolve_canvas_size,
)
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam


REFERENCE_RESIZE_MODES = ("match", "max", "diffusers")
DEFAULT_REFERENCE_RESIZE_MODE = "match"
REFERENCE_SHORT_EDGE = 2048


def resolve_reference_image_size(
    width: int,
    height: int,
    *,
    target_width: int,
    target_height: int,
    mode: str = DEFAULT_REFERENCE_RESIZE_MODE,
    multiple: int = 32,
    max_short_edge: int = REFERENCE_SHORT_EDGE,
) -> tuple[int, int]:
    """Return ``(height, width)`` for one of the three reference policies."""
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"The target canvas must have a positive size, got {target_width}x{target_height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    if mode not in REFERENCE_RESIZE_MODES:
        raise ValueError(f"reference_resize_mode must be one of {REFERENCE_RESIZE_MODES}, got {mode!r}.")

    if mode == "match":
        scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    elif mode == "max":
        scale = min(1.0, max_short_edge / min(width, height))
    else:
        scale = max_short_edge / min(width, height)

    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def _reference_image_to_pil(components, image) -> Image.Image:
    """Apply the same accepted-layout conversion as Diffusers' setup block."""
    if isinstance(image, torch.Tensor):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        image = components.image_processor.pt_to_numpy(image[None])[0]
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"A reference image must be `(height, width, 3)` RGB pixels, got {tuple(image.shape)}.")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = components.image_processor.numpy_to_pil(image)[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported reference image type: {type(image)}.")
    return image.convert("RGB")


class MiniMaxH3Ref2VAResizeSetupStep(MiniMaxH3Ref2VASetupStep):
    """Stock Ref2VA setup with a selectable image-only resize policy."""

    @property
    def inputs(self):
        return [
            *super().inputs,
            InputParam(
                name="reference_resize_mode",
                type_hint=str,
                default=DEFAULT_REFERENCE_RESIZE_MODE,
                description=(
                    "Reference image resize policy: `match` follows the target pixel area, `max` caps the short "
                    "edge at 2048 without upscaling, and `diffusers` forces a 2048-pixel short edge."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state):
        block_state = self.get_block_state(state)

        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        multiple = components.canvas_multiple
        if block_state.height is not None and (block_state.height % multiple or block_state.width % multiple):
            raise ValueError(
                f"`height` and `width` must be multiples of {multiple}, got "
                f"{block_state.height}x{block_state.width}."
            )
        if block_state.reference_resize_mode not in REFERENCE_RESIZE_MODES:
            raise ValueError(
                f"reference_resize_mode must be one of {REFERENCE_RESIZE_MODES}, "
                f"got {block_state.reference_resize_mode!r}."
            )
        if not block_state.references:
            raise ValueError("`ref2va` needs at least one reference; use the `t2va` workflow for text-only requests.")
        for index, entry in enumerate(block_state.references):
            if not isinstance(entry, MiniMaxH3Reference):
                raise ValueError(
                    f"`references[{index}]` must be a MiniMax-H3 image, video or audio reference, got {type(entry)}."
                )
        kinds = [entry.kind for entry in block_state.references]
        for kind, limit in (("image", self.max_images), ("video", self.max_videos), ("audio", self.max_audios)):
            if kinds.count(kind) > limit:
                raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
        if len(kinds) > self.max_references:
            raise ValueError(
                f"MiniMax-H3 accepts at most {self.max_references} references in total, got {len(kinds)}."
            )
        if set(kinds) == {"audio"}:
            raise ValueError("An audio reference cannot be used without at least one image or video reference.")

        if block_state.height is None:
            block_state.height, block_state.width = resolve_canvas_size(
                16,
                9,
                multiple,
                components.config.canvas_short_edge,
                components.config.canvas_max_pixels,
            )
        aligned_num_frames = align_num_frames(
            block_state.num_frames,
            components.vae_frames_per_chunk,
            components.vae_latents_per_chunk,
        )
        duration = aligned_num_frames / components.fps
        if not components.min_duration <= duration <= components.max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {components.min_duration} and {components.max_duration} seconds at "
                f"{components.fps} fps, got {block_state.num_frames} frames "
                f"(aligned to {aligned_num_frames})."
            )
        if aligned_num_frames != block_state.num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5; rounding {block_state.num_frames} "
                f"up to {aligned_num_frames}."
            )
        block_state.num_frames = aligned_num_frames

        normalized = []
        for entry in block_state.references:
            waveform = None
            if entry.has_audio:
                sample_rate = entry.sample_rate or components.audio_sampling_rate
                waveform = self._normalize_audio_condition(
                    entry.audio,
                    sample_rate,
                    components.audio_sampling_rate,
                    max_duration=block_state.num_frames / components.fps,
                )

            if entry.kind == "image":
                image = _reference_image_to_pil(components, entry.image)
                target_height, target_width = resolve_reference_image_size(
                    *image.size,
                    target_width=block_state.width,
                    target_height=block_state.height,
                    mode=block_state.reference_resize_mode,
                    multiple=multiple,
                    max_short_edge=components.config.reference_image_short_edge,
                )
                if image.size != (target_width, target_height):
                    image = components.image_processor.resize(image, height=target_height, width=target_width)
                normalized.append(MiniMaxH3ImageReference(image=image))
            elif entry.kind == "video":
                normalized.append(
                    MiniMaxH3VideoReference(
                        frames=self._normalize_video_condition(
                            entry.frames,
                            float(entry.fps),
                            block_state.num_frames,
                            multiple,
                            components.config.canvas_short_edge,
                            components.config.canvas_max_pixels,
                            float(components.fps),
                        ),
                        fps=float(components.fps),
                        audio=waveform,
                        sample_rate=None if waveform is None else components.audio_sampling_rate,
                    )
                )
            else:
                normalized.append(
                    MiniMaxH3AudioReference(audio=waveform, sample_rate=components.audio_sampling_rate)
                )
        block_state.normalized_references = normalized
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ResizeAutoBeforeEncodeStep(MiniMaxH3AutoBeforeEncodeStep):
    """Replace only the Ref2VA option of Diffusers' conditional setup block."""

    block_classes = [MiniMaxH3Ref2VAResizeSetupStep, *MiniMaxH3AutoBeforeEncodeStep.block_classes[1:]]


class MiniMaxH3Ref2VAResizeBlocks(MiniMaxH3Blocks):
    """Stock MiniMax-H3 blocks with the custom before-encode conditional."""

    block_classes = [MiniMaxH3ResizeAutoBeforeEncodeStep, *MiniMaxH3Blocks.block_classes[1:]]


class MiniMaxH3Ref2VAResizePipeline(MiniMaxH3ModularPipeline):
    """MiniMax-H3 pipeline whose Ref2VA image policy defaults to ``match``."""

    pass


def load_minimax_h3_ref2va_pipeline(model_id, *, components_manager=None):
    """Build the local blocks while preserving Diffusers' component loading."""
    return MiniMaxH3Ref2VAResizePipeline(
        blocks=MiniMaxH3Ref2VAResizeBlocks(),
        pretrained_model_name_or_path=model_id,
        components_manager=components_manager,
    )
