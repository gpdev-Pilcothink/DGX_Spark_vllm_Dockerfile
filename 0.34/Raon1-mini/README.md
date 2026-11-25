# Raon-1-mini NVFP4 + DFlash2 NVFP4 on a Single DGX Spark

Serve KRAFTON's Raon-1-mini NVFP4 target with **Pilcothink's NVFP4 DFlash2 draft** on one NVIDIA GB10. The Spark image includes vLLM 0.34.0 and FlashInfer 0.7.3 with B12X. The draft proposes four tokens per step and uses its own quantized output head.

## Prerequisites

- One DGX Spark with Linux ARM64, Docker and NVIDIA GPU support.
- The Hugging Face CLI (`hf`) and local storage for both checkpoints.

## Models

| Role | Checkpoint | How the server uses it |
|---|---|---|
| Target | [KRAFTON/Raon-1-mini-nvfp4](https://huggingface.co/KRAFTON/Raon-1-mini-nvfp4) | Provides the tokenizer and input embeddings, and verifies the draft tokens. |
| DFlash2 draft | [Pilcothink/Raon-1-mini-dflash2-nvfp4](https://huggingface.co/Pilcothink/Raon-1-mini-dflash2-nvfp4) | Uses NVFP4 W4A4 for 36 body projections and its own `lm_head`. |

## Step 1. Download the Models

Replace `/path/to/models` throughout this guide with an absolute directory on your Spark. Download the full draft repository so its index can resolve both Safetensors files.

```sh
hf download KRAFTON/Raon-1-mini-nvfp4 \
  --revision f6962d7a7d498fbb32347514e5f254b1b577b7b8 \
  --local-dir /path/to/models/Raon-1-mini-nvfp4

hf download Pilcothink/Raon-1-mini-dflash2-nvfp4 \
  --revision 5cc424aa4146e12f0485c5250292346dad1ffe12 \
  --local-dir /path/to/models/Raon-1-mini-dflash2-nvfp4
```

## Step 2. Pull the Docker Image

```sh
docker pull pilcothink/vllm_spark:raon-1-mini
```

## Step 3. Configure DGX Spark

The server reads both model directories through read-only bind mounts. If the checkpoints are already on the Spark, point each `src=` below to the directory containing its `config.json`.

| Checkpoint | Spark directory in this example | Container directory |
|---|---|---|
| Target | `/path/to/models/Raon-1-mini-nvfp4` | `/models/target` |
| Draft | `/path/to/models/Raon-1-mini-dflash2-nvfp4` | `/models/draft` |

## Start the Server

```sh
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --hostname raon-spark \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src=/path/to/models/Raon-1-mini-nvfp4,dst=/models/target,readonly \
  --mount type=bind,src=/path/to/models/Raon-1-mini-dflash2-nvfp4,dst=/models/draft,readonly \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  pilcothink/vllm_spark:raon-1-mini \
  /models/target \
    --served-model-name Raon-1-mini \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --load-format safetensors \
    --tensor-parallel-size 1 \
    --decode-context-parallel-size 1 \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --attention-backend B12X_MLA \
    --attention-config '{"flash_attn_version":2,"mla_prefill_backend":"FLASH_ATTN"}' \
    --kv-cache-dtype fp8_e4m3 \
    --kv-cache-memory-bytes 60000000000 \
    --gpu-memory-utilization 0.75 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --max-model-len 262144 \
    --max-num-seqs 100 \
    --max-num-batched-tokens 2048 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"dflash","model":"/models/draft","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","attention_backend":"FLASH_ATTN","kv_cache_dtype":"auto","moe_backend":"auto"}' \
    --reasoning-parser raon_1 \
    --tool-call-parser raon_1 \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
```

## Notes

| Operation | Target | DFlash2 draft |
|---|---|---|
| Linear weights and compute | B12X uses the checkpoint's per-layer precision. Routed MoE also uses B12X. | B12X NVFP4 W4A4 for the body projections and output head. |
| Attention | B12X MLA for decode and verification, FlashAttention for MLA prefill. RDA layers use the Raon implementation. | FlashAttention 2 for non-causal draft attention. |
| KV cache | FP8 E4M3 | BF16 (`auto` for this draft) |

The explicit 60,000,000,000-byte KV budget takes precedence over utilization-based cache sizing. The 262,144-token limit applies to one sequence. The number of long requests that fit concurrently depends on their lengths and the available KV capacity.

## Verify Server

After startup completes, run these commands from another terminal on the Spark.

```sh
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/v1/models
```

The model list should contain `Raon-1-mini`. Send a short request to verify generation.

```sh
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Raon-1-mini","messages":[{"role":"user","content":"한국의 수도는 어디인가요?"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

## ===========Test===========

The draft export passed the structure check for all 37 NVFP4 Linear groups. The output head was calibrated using activations from the exported NVFP4 body.

## Sources and Credits

- KRAFTON AI released the [NVFP4 target](https://huggingface.co/KRAFTON/Raon-1-mini-nvfp4) and the [original DFlash2 draft](https://huggingface.co/KRAFTON/Raon-1-mini-dflash2).
- Pilcothink published the [quantized DFlash2 draft](https://huggingface.co/Pilcothink/Raon-1-mini-dflash2-nvfp4) and the [DGX Spark serving image](https://hub.docker.com/r/pilcothink/vllm_spark/tags?name=raon-1-mini).
- The conversion uses [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/6c475d716cff1c9c70b868ae4da86a59ccb1aa99). Serving uses [vLLM 0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
