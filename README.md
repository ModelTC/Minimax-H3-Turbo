# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides batch MiniMax-H3 inference and NFE/LoRA comparisons.

## Model specs

| Model | Tasks | Training resolution | Training shifts (video / audio) | Distillation steps (NFE) | Recommended inference steps (NFE) |
| --- | --- | --- | ---: | ---: | ---: |
| [FL2VA Turbo 4-step v0.1](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors) | FL2VA / T2VA | 544p (mixed aspect ratio) | 12 / 3 | 4 | 4 |
| [FL2VA Turbo 8-step v1.0 (Diffusers BF16)](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors)<br>[FL2VA Turbo 8-step v1.0 (ComfyUI BF16)](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | FL2VA / T2VA | 544p (mixed aspect ratio) | 12 / 3 | 8 | 8 |

For `NFE = N`, define the N transformer evaluation points on the unshifted grid as
`q_i = (N - i) / N`, where `i = 0, 1, ..., N - 1`.

For example, with `NFE = 4`, `video shift = 12`, and `audio shift = 3`, the shared
grid is `q = [1, 0.75, 0.5, 0.25]`, giving video sigma
`[1, 0.9730, 0.9231, 0.8000] -> 0` and audio sigma
`[1, 0.9000, 0.7500, 0.5000] -> 0`; each list therefore uses exactly four NFEs.

## Download the LoRA checkpoints

Download the [4-step v0.1](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors) and [8-step v1.0](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors) checkpoints to the repository root:

```bash
python -m pip install -U huggingface_hub
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
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
