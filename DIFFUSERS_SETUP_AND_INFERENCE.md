# Diffusers Setup and Inference

## Environment

Follow the [Diffusers MiniMax-H3 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
to prepare a compatible Diffusers version and Python environment. The inference
script uses `MiniMaxAI/MiniMax-H3` as the base model by default.

## Download LoRA checkpoints

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

Each test example specifies `duration`, `megapixels`, and `aspect_ratio`.
The inference script resolves the final width and height from these fields
using [`resolution_util.py`](resolution_util.py) and rounds both dimensions to
multiples of 32. Supported aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, and
their transposes.

## Multi-GPU inference with FSDP2

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

To run the base model with FSDP2, omit `--lora-path`. FSDP2 shards only the
text encoder and the active transformer; the remaining pipeline components are
replicated on every rank.

## Single-GPU inference

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

Ref2VA, 4 NFE (v0.1):

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_ref2va_test.json \
  --lora-path minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --inference-steps 4 \
  --output-dir outputs/ref2va_lora_4nfe
```

Ref2VA base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json examples/prompts_ref2va_test.json \
  --inference-steps 50 \
  --output-dir outputs/ref2va_base_50nfe
```

The Ref2VA jobs JSON must contain only `ref2va` examples. The script selects
`transformer_ref` automatically. Do not use an FL2VA LoRA checkpoint for this
command unless it was specifically trained for `transformer_ref`.

LoRA, 4 NFE, 768p (v1.0):

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

To fuse LoRA weights into the model before inference, add `--fuse-lora`.
