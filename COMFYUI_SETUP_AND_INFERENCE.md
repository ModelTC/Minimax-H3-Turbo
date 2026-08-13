# ComfyUI Setup and Inference

## Environment

**ComfyUI 0.31.0 or newer is required.** The example graphs use ComfyUI core
nodes only, including the built-in MiniMax-H3 subgraph
(`MiniMaxH3ImageToVideo`).

## Required models

Download the base MiniMax-H3 components from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)
and place them in the corresponding ComfyUI model directories.

Download the ComfyUI LoRAs from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
into `ComfyUI/models/loras/`:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors \
  --local-dir ComfyUI/models/loras
```

```text
ComfyUI/
└── models/
    └── loras/
        ├── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
        ├── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
        └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

## Example workflows

Ready-to-import graphs are listed in the main [README](README.md). The T2VA
and I2VA graphs default to **FL2VA Turbo 8-step v1.0**; the Ref2VA graph uses
**Ref2VA Turbo 4-step v0.1**.

## Inputs

On the subgraph, set `lora_name`, `steps`, `shift_video`, and `shift_audio`
together to match the model table. Sampler defaults to `euler`.

**Resolution.** Width and height must be multiples of 32. Change the aspect
ratio or megapixel value on **Resolution Selector** when another size is needed.
The I2VA graph also includes an unused **Use Image Size** group
(`ImageScaleToTotalPixels` → `GetImageSize`); wire those outputs to the
subgraph `width` / `height` if you want the video canvas to follow the input
image instead of the selector.

**Duration.** Defaults to `5` seconds. MiniMax-H3 runs at 24 FPS and the frame
count must be `17 * n + 5`, so a 5-second request is snapped to 124 frames
(about 5.17 seconds).

**Prompts.** Use three sections:

```text
integrated_multimodal_description: [Shot 1] Describe the visual style, subject, action, camera, lighting, and dialogue.

overall_soundscape: Describe dialogue, ambient sound, and synchronized sound effects.

non_diegetic_music: Describe the background score, or use N/A for no score.
```

Dialogue can be written as `<d>[English] Dialogue text.</d>`.

For I2VA / FL2VA, keep first-frame identity with `<Picture 1>` (and
`<Picture 2>` if `last_frame` is connected):

```text
integrated_multimodal_description: For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

[Shot 1] Preserve the subject, clothing, and scene from <Picture 1>, then describe the motion.
```

## Run

Drag a workflow JSON onto the ComfyUI canvas, set the inputs above, and queue
it. Output is written under `video/MiniMax_H3`.
