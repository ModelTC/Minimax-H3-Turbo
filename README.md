# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides batch MiniMax-H3 inference and NFE/LoRA comparisons.

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
  </tbody>
</table>


For `NFE = N`, define the N transformer evaluation points on the unshifted grid as
`q_i = (N - i) / N`, where `i = 0, 1, ..., N - 1`.

For example, with `NFE = 4`, `video shift = 12`, and `audio shift = 3`, the shared
grid is `q = [1, 0.75, 0.5, 0.25]`, giving video sigma
`[1, 0.9730, 0.9231, 0.8000] -> 0` and audio sigma
`[1, 0.9000, 0.7500, 0.5000] -> 0`; each list therefore uses exactly four NFEs.

## Download the LoRA checkpoints

Download the [4-step v0.1](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors), [8-step v1.0](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors), and [4-step v1.0 768p](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors) checkpoints to the repository root:

```bash
python -m pip install -U huggingface_hub
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --local-dir .
```



## Environment setup

Follow the [Diffusers MiniMax-H3 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) to prepare a compatible Diffusers version and Python environment.

## Run inference

The inference code uses `MiniMaxAI/MiniMax-H3` as the base model by default.

### Multi-GPU inference with FSDP2

FSDP2 shards the text encoder and the active transformer across GPUs, so CPU
offload is disabled. Launch one process per GPU with `torchrun` (PyTorch >= 2.6
and an NCCL-capable environment are required):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 \
  inference_minimax_h3.py \
  --fsdp2 \
  --jobs-json prompts_t2va_test_24.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe_fsdp2
```

To run the base model with FSDP2, omit `--lora-path`. FSDP2 shards only the text
encoder and the active transformer; the remaining pipeline components are
replicated on every rank.

### Single-GPU inference

LoRA, 4 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_24.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --inference-steps 4 \
  --output-dir outputs/lora_4nfe
```

LoRA, 8 NFE (v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_24.json \
  --lora-path minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
  --inference-steps 8 \
  --output-dir outputs/lora_8nfe
```

LoRA, 4 NFE, 768p(v1.0):

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_24.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --inference-steps 4 \
  --video-shift 6 \
  --lora-alpha 128 \
  --height 768 \
  --width 1344 \
  --output-dir outputs/lora_4nfe_768p
```

Base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_24.json \
  --inference-steps 50 \
  --output-dir outputs/base_50nfe
```

To fuse the LoRA weights into the model before inference, add the following option to the LoRA command:

```bash
--fuse-lora
```



## Roadmap

1. Improve the visual details and overall quality of FL2V Turbo.
2. Develop distillation based on Ref2V.
