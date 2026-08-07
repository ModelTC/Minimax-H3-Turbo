# [Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)

Minimax-H3-Turbo provides batch MiniMax-H3 inference and NFE/LoRA comparisons.

> [!NOTE]
> [v0.1 is a preview release](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/2). Its current visual quality can be reviewed in the linked examples, and the image details still need improvement. An enhanced version is in development.

## Download the LoRA checkpoint

First, download [minimax_h3_fl2v_turbo_4step_v0.1.safetensors](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors) to the repository root:

```bash
python -m pip install -U huggingface_hub
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --local-dir .
```

## Environment setup

Follow the [Diffusers MiniMax-H3 documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) to prepare a compatible Diffusers version and Python environment.

## Run inference

The inference code uses `MiniMaxAI/MiniMax-H3` as the base model by default.

LoRA, 4 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_20.json \
  --lora-path minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --inference-steps 4 \
  --output-dir outputs/lora_4nfe
```

Base model, 50 NFE:

```bash
python inference_minimax_h3.py \
  --jobs-json prompts_t2va_test_20.json \
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
