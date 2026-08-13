# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides MiniMax-H3 Turbo LoRA checkpoints, plus Diffusers
batch inference and ComfyUI workflows.

## Model specs

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th align="center">Tasks</th>
      <th align="center">Training<br>resolution</th>
      <th align="center">Training shifts<br>(video / audio)</th>
      <th align="center">Distillation<br>steps (NFE)</th>
      <th align="center">Recommended inference<br>steps (NFE)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <strong>FL2VA Turbo 4-step v0.1</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
    <tr>
      <td>
        <strong>FL2VA Turbo 8-step v1.0</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">8</td>
      <td align="center">8 / 4</td>
    </tr>
    <tr>
      <td>
        <strong>FL2VA Turbo 4-step v1.0 768p</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">FL2VA / T2VA</td>
      <td align="center">768p<br><sub>1344x768</sub></td>
      <td align="center">6 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
    <tr>
      <td>
        <strong>Ref2VA Turbo 4-step v0.1</strong><br>
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors">Diffusers</a> ·
        <a href="https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors">ComfyUI</a>
      </td>
      <td align="center">Ref2VA</td>
      <td align="center">544p<br><sub>mixed aspect ratio</sub></td>
      <td align="center">12 / 3</td>
      <td align="center">4</td>
      <td align="center">4</td>
    </tr>
  </tbody>
</table>


### Note on shift

For `NFE = N`, define the N transformer evaluation points on the unshifted grid as
`q_i = (N - i) / N`, where `i = 0, 1, ..., N - 1`.

For example, with `NFE = 4`, `video shift = 12`, and `audio shift = 3`, the shared
grid is `q = [1, 0.75, 0.5, 0.25]`, giving video sigma
`[1, 0.9730, 0.9231, 0.8000] -> 0` and audio sigma
`[1, 0.9000, 0.7500, 0.5000] -> 0`; each list therefore uses exactly four NFEs.

### Note on reference-image resizing

The three reference-image resizing policies used by our workflows are based on
the `ref_image_size` implementation described in [ComfyUI's MiniMax H3 R2V reference-image sizing guidance](https://docs.comfy.org/tutorials/video/minimax/minimax-h3#prompting-tips-3):

<table>
  <thead>
    <tr>
      <th align="center">Mode</th>
      <th>Behavior</th>
      <th align="center">Scale factor<br><sub>before 32-pixel rounding</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><code>match</code></td>
      <td>Matches the reference pixel area to the target canvas while preserving the reference aspect ratio. It never upscales a smaller reference.</td>
      <td align="center"><code>min(1, sqrt(target_area / ref_area))</code></td>
    </tr>
    <tr>
      <td align="center"><code>max</code></td>
      <td>Preserves the reference aspect ratio and only scales down references whose short edge exceeds 2048 pixels.</td>
      <td align="center"><code>min(1, 2048 / ref_short_edge)</code></td>
    </tr>
    <tr>
      <td align="center"><code>diffusers</code></td>
      <td>Preserves the reference aspect ratio and forces the short edge to 2048 pixels, matching the original Diffusers behavior.</td>
      <td align="center"><code>2048 / ref_short_edge</code></td>
    </tr>
  </tbody>
</table>

All three policies keep the reference aspect ratio, use the H3 resolution grid
(dimensions rounded to multiples of 32), and avoid cropping the reference
content. In our distillation training, we use `match`, so the reference-image
pixel budget follows the target training resolution.

The Ref2VA inference entry point in this repository exposes the same three
policies through `--reference-resize-mode` and defaults to `match`. Passing
`--reference-resize-mode diffusers` restores the original Diffusers behavior
(the fixed 2048-pixel short edge). For our distilled models, **we recommend
selecting `match`** so inference uses the same resizing policy as training.

## Diffusers inference

### Environment

Follow the [Diffusers MiniMax-H3 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
to prepare a compatible Diffusers version and Python environment. The inference
script uses `MiniMaxAI/MiniMax-H3` as the base model by default.

### Download LoRA checkpoints

Download the Diffusers checkpoints from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
to the repository root:

```bash
python -m pip install -U huggingface_hub
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --local-dir .
```

Jobs live in `examples/`:

- T2VA: [`examples/prompts_t2va_test.json`](examples/prompts_t2va_test.json)
- I2VA: [`examples/prompts_i2va_test.json`](examples/prompts_i2va_test.json)
- Ref2VA: [`examples/prompts_ref2va_test.json`](examples/prompts_ref2va_test.json)

Pass `--jobs-json` and `--lora-path` as in the commands below; the script fills
in resolution, duration, and prompt fields from the JSON.

### Run

#### Multi-GPU inference with FSDP2

FSDP2 shards the text encoder and the active transformer across GPUs, so CPU
offload is disabled. Launch one process per GPU with `torchrun` (PyTorch >= 2.6
and an NCCL-capable environment are required):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 \
  inference_minimax_h3.py \
  --fsdp2 \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe_fsdp2
```

To run the base model with FSDP2, omit `--lora-path`. FSDP2 shards only the text
encoder and the active transformer; the remaining pipeline components are
replicated on every rank.

#### Single-GPU inference

LoRA, 4 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --inference-steps 4 \
  --output-dir outputs/lora_4nfe
```

LoRA, 8 NFE (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe
```

I2VA, 8 NFE (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_i2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/i2va_lora_8nfe
```

Ref2VA base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_ref2va_test.json \
  --inference-steps 50 \
  --output-dir outputs/ref2va_base_50nfe
```

The Ref2VA jobs JSON must contain only `ref2va` examples. The script selects
`transformer_ref` automatically. Do not pass the FL2VA LoRA checkpoints above
to this command unless the checkpoint was specifically trained for
`transformer_ref`.

Each test example specifies `duration`, `megapixels`, and `aspect_ratio`.
The inference script resolves the final width and height from these fields
using the shared [`resolution_util.py`](resolution_util.py) table and rounds
both dimensions to multiples of 32. Supported aspect ratios are `21:9`,
`16:9`, `4:3`, `1:1`, and their transposes.

LoRA, 4 NFE, 768p(v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --inference-steps 4 \
  --video-shift 6 \
  --lora-alpha 128 \
  --megapixels 1.0 \
  --aspect-ratio 16:9 \
  --output-dir outputs/lora_4nfe_768p
```

Base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_t2va_test.json \
  --inference-steps 50 \
  --output-dir outputs/base_50nfe
```

To fuse the LoRA weights into the model before inference, add `--fuse-lora`.

## ComfyUI inference

### Environment

**ComfyUI 0.31.0 or newer is required.** The example graphs use ComfyUI core
nodes only, including the built-in MiniMax-H3 subgraph
(`MiniMaxH3ImageToVideo`).

### Required models

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

### Example workflows

Ready-to-import graphs are in [example_workflows](example_workflows/). The T2VA
and I2VA graphs default to **FL2VA Turbo 8-step v1.0**; the Ref2VA graph uses
**Ref2VA Turbo 4-step v0.1**.

| Workflow | Task | Default resolution |
|---|---|---|
| [video_minimax_h3_t2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_t2v_lightx2v_turbo.json) | T2VA (text-to-video + audio) | 960×544 (`16:9`, `0.5` MP) |
| [video_minimax_h3_i2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_i2v_lightx2v_turbo.json) | I2VA / FL2VA (image-to-video + audio) | 864×480 (`16:9`, `0.4` MP) |
| [video_minimax_h3_ref2v_lightx2v_turbo.json](example_workflows/video_minimax_h3_ref2v_lightx2v_turbo.json) | Ref2VA (reference-to-video + audio) | 960×544 (`16:9`, `0.5` MP) |

All graphs wrap the same MiniMax-H3 subgraph. T2VA leaves `first_frame` /
`last_frame` unconnected; I2VA connects a `LoadImage` to `first_frame`, with
`last_frame` optional for first/last-frame interpolation. Ref2VA connects one
or more reference images through the reference-input branch.

### Inputs

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

### Run

Drag a workflow JSON onto the ComfyUI canvas, set the inputs above, and queue
it. Output is written under `video/MiniMax_H3`.

## Roadmap

1. Improve the visual quality and consistency of Ref2VA and FL2VA Turbo.

## Acknowledgements

Some Ref2VA test cases and reference assets are adapted from public showcases on
the [Hailuo website](https://hailuoai.video/) and from the [MiniMax-H3 discussion
on Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/29).
We thank the community contributors for sharing their test assets and prompts.

Special thanks to the
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) project and the
MiniMax team for open-sourcing the MiniMax-H3 model.
