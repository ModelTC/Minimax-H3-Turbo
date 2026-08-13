# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides MiniMax-H3 Turbo LoRA checkpoints, plus Diffusers
batch inference and ComfyUI workflows.

## 1. Model specs

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

## 2. Diffusers setup and inference

See [DIFFUSERS_SETUP_AND_INFERENCE.md](DIFFUSERS_SETUP_AND_INFERENCE.md) for
environment setup, checkpoint downloads, test JSON files, and single- or
multi-GPU inference commands.

## 3. ComfyUI inference

See [COMFYUI_SETUP_AND_INFERENCE.md](COMFYUI_SETUP_AND_INFERENCE.md) for
ComfyUI requirements, model installation, inputs, prompts, and run instructions.

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

For detailed workflow inputs and execution steps, see
[COMFYUI_SETUP_AND_INFERENCE.md](COMFYUI_SETUP_AND_INFERENCE.md).

## 4. Roadmap

1. Improve the visual quality and consistency of Ref2VA and FL2VA Turbo.

## 5. Acknowledgements

Some Ref2VA test cases and reference assets are adapted from public showcases on
the [Hailuo website](https://hailuoai.video/) and from the [MiniMax-H3 discussion
on Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/29).
We thank the community contributors for sharing their test assets and prompts.

Special thanks to the
[MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) project and the
MiniMax team for open-sourcing the MiniMax-H3 model.
