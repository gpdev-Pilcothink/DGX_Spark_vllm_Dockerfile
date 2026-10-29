# Raon-1-mini-preview NVFP4 + B12X on DGX Spark — v2

This recipe runs **KRAFTON Raon-1-mini-preview NVFP4** on **one DGX Spark** using **vLLM 0.32.0**, **B12X**, and the official **Raon DFlash2 draft model**.

The serving configuration uses **262,144 tokens of context**, **4 speculative tokens**, **probabilistic drafting with block verification**, and **async scheduling**.

B12X handles the target model's Linear and MoE operations and MLA decode/verification. MLA prefill uses FlashAttention, and the RDA layers retain the Raon implementation.

---

## Prerequisites

* One **DGX Spark / NVIDIA GB10**, Linux ARM64, SM121.
* A working NVIDIA driver and Docker GPU runtime compatible with the image's CUDA 13.0 stack.
* The Hugging Face CLI (`hf`) for downloading both models.
* Local storage for the Docker image, model files, and persistent compilation cache.

This is a **single-node, TP1/DCP1** recipe. It runs directly with `docker run`; no cluster helper is required.

---

## Models

### Target Model

**KRAFTON/Raon-1-mini-preview-nvfp4**

[Hugging Face model repository](https://huggingface.co/KRAFTON/Raon-1-mini-preview-nvfp4)

### DFlash2 Draft Model

**KRAFTON/Raon-1-mini-preview-dflash2**

[Hugging Face draft repository](https://huggingface.co/KRAFTON/Raon-1-mini-preview-dflash2)

Both models are required for the serving command below. Download the complete repositories, including their configuration and tokenizer files.

---

## Step 1. Download the Models

Replace `/path` with your own storage directory.

```bash
hf download KRAFTON/Raon-1-mini-preview-nvfp4 \
  --local-dir /path/Model/Raon-1-mini-preview-nvfp4

hf download KRAFTON/Raon-1-mini-preview-dflash2 \
  --local-dir /path/Model/Raon-1-mini-preview-dflash2
```

Example directory structure:

```text
/path/
├── Model/
│   ├── Raon-1-mini-preview-nvfp4/
│   └── Raon-1-mini-preview-dflash2/
└── raon_cache/
```

---

## Step 2. Pull Docker Image

<!-- Release preparation: publish this image tag before publishing the pull guide. -->

```bash
docker pull pilcothink/vllm_spark:0.32-Raon-1-v2
```

---

## Step 3. Configure DGX Spark

```bash
export VLLM_IMAGE=pilcothink/vllm_spark:0.32-Raon-1-v2
export HOST_AI_DIR=/path
export CACHE_DIR=/path/raon_cache

mkdir -p "$CACHE_DIR"
```

Replace both host paths with values from your environment. The model storage directory is mounted at `/workspace` inside the container. The persistent cache directory is mounted at `/root/.cache`.

---

# DFlash2 Serving

The default configuration uses **262,144 tokens**, **4 speculative tokens**, and **up to 8 concurrent sequences**, subject to available KV capacity.

Run on the **DGX Spark**:

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  --hostname raon-spark \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOST_AI_DIR:/workspace" \
  -v "$CACHE_DIR:/root/.cache" \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  --entrypoint vllm \
  "$VLLM_IMAGE" \
  serve /workspace/Model/Raon-1-mini-preview-nvfp4 \
    --served-model-name Raon-1-mini-preview \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --decode-context-parallel-size 1 \
    --dtype bfloat16 \
    --load-format safetensors \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --attention-backend B12X_MLA \
    --attention-config '{"flash_attn_version":2}' \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.75 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"dflash","model":"/workspace/Model/Raon-1-mini-preview-dflash2","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","attention_backend":"FLASH_ATTN","kv_cache_dtype":"auto","moe_backend":"auto"}' \
    --reasoning-parser raon_1 \
    --tool-call-parser raon_1 \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"enable_thinking":true,"truncate_history_thinking":false}'
```

## DFlash2 Draft Length

The reference command uses:

```json
"num_speculative_tokens": 4
```

This is a serving setting, not a fixed value baked into the Docker image. Changing it requires restarting the server and preparing the corresponding CUDA graphs. It does not require rebuilding the image or editing the draft checkpoint.

The Raon configuration in this image sets DFlash drafting to `probabilistic`. Omitting `draft_sample_method` or specifying `greedy` does not override that model-specific setting.

`rejection_sample_method` is a separate setting. The example uses `block`; `standard` is also available. Block verification changes how proposed tokens are accepted, not the KV cache block size. Compare output throughput under the same workload when selecting a verification method.

## CUDA Graph Capture Sizes

The recipe uses **V2 Model Runner** and:

```json
{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}
```

`mode: 0` disables torch.compile/Inductor compilation; it does not disable the explicitly selected CUDA graphs. Both target decode/verification and the DFlash2 draft model use FULL graph capture in this configuration.

With four speculative tokens, a regular target verification step uses **five token rows per request**: four draft tokens plus one target token. At eight active requests, that is up to **40 rows**. These are execution rows, not context tokens.

The command leaves capture-size selection to this image's runner. Changing draft length or concurrency can change the selected graph sizes and memory use.

## Memory Tuning Order

The reference settings are:

```bash
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.75 \
```

1. Choose the draft length and maximum active sequences first.
2. Check the startup KV capacity after graph preparation. The context limit includes the prompt and generated output.
3. If KV capacity is insufficient and the system has memory headroom, increase the utilization target conservatively and restart.
4. If an actual out-of-memory error occurs, reduce the memory target and/or concurrency, and lower the context limit if the resulting KV capacity requires it.

The command does not fix a KV byte budget. Eight serving slots do not guarantee eight simultaneous full 262,144-token contexts. Leave memory available for the operating system and other applications on Spark's shared-memory platform.

---

## Notes

* This recipe targets **one DGX Spark**, with **TP1/DCP1**.
* `B12X_MLA` is a **custom backend registered by this image**, connecting Raon's MLA decode/verification to the existing B12X `dense_mla` kernels. It is not a stock official-vLLM backend name.
* The target uses FP8 E4M3 KV. The draft uses FlashAttention 2 and `kv_cache_dtype: "auto"`, which resolves to BF16 for this draft; it does not inherit target FP8 KV.
* Checkpoint-specific W4A4/W4A16 declarations are preserved. A W4A16 Linear log does not mean every routed expert uses W4A16.
* Raon's existing FP8 query and q/KV scales are passed to B12X MLA. Split partials use FP32. Different kernels can produce numerical differences; this guide does not claim bitwise equivalence or a measured quality score.
* The image includes the Raon DFlash cache-alignment fixes and B12X compiled-program lifetime retention for FULL graphs.
* Keep `/root/.cache` persistent. First use of a new shape can trigger JIT compilation or B12X preparation and temporarily increase latency. Warm up before measuring throughput.
* The recipe retains the model's generation defaults unless the client overrides them. Use identical sampling settings for performance comparisons.
* Startup and subsequent generation have been observed on Spark with the v2 image. This does not establish performance or quality across every context length and concurrency setting.

Implementation references:

* [Official vLLM 0.32 source](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)
* [KRAFTON Raon implementation](https://github.com/krafton-ai/vllm/tree/fec32a3e39d51ea71afeef7c83ca572cdcab2f0b)
* [eugr Spark Docker reference](https://github.com/eugr/spark-vllm-docker/tree/b62168ed4eb5dd8d788274e7a1677e7ce271bf5b)
* [Pinned FlashInfer B12X source](https://github.com/local-inference-lab/flashinfer/tree/0b6178857791d038e24fa8efb0ad04d55b6b5ccc)
* [B12X MLA integration reference](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/v1/attention/backends/mla/b12x_mla.py)

---

## Verify Server

Replace `<SPARK_IP>` with the reachable address of your Spark.

```bash
curl "http://<SPARK_IP>:8000/v1/models"
```

The OpenAI-compatible API is available at:

```text
http://<SPARK_IP>:8000/v1
```

Use **`Raon-1-mini-preview`** as the API model name. A model-list response confirms API availability; verify actual generation with a completion request as well.

```bash
curl "http://<SPARK_IP>:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Raon-1-mini-preview","messages":[{"role":"user","content":"Explain what a GPU does."}],"max_tokens":256,"stream":true}'
```

---

## ===========Test===========
```
| model               |           test |              t/s |       peak t/s |      ttfr (ms) |   est_ppt (ms) |   e2e_ttft (ms) |
|:--------------------|---------------:|-----------------:|---------------:|---------------:|---------------:|----------------:|
| Raon-1-mini-preview |         pp2048 |  6699.17 ± 93.82 |                |  314.29 ± 4.22 |  304.57 ± 4.22 |   315.11 ± 4.20 |
| Raon-1-mini-preview |          tg128 |     67.68 ± 3.71 |   76.00 ± 5.72 |                |                |                 |
| Raon-1-mini-preview |         pp2048 |  6697.94 ± 73.51 |                |  314.32 ± 3.37 |  304.61 ± 3.37 |   314.83 ± 3.59 |
| Raon-1-mini-preview |         tg1024 |     62.60 ± 5.47 | 101.33 ± 12.47 |                |                |                 |
| Raon-1-mini-preview | pp2048 @ d1024 | 6020.83 ± 103.49 |                |  517.93 ± 8.69 |  508.22 ± 8.69 |   519.36 ± 8.23 |
| Raon-1-mini-preview |  tg128 @ d1024 |     62.24 ± 4.66 |   73.00 ± 3.56 |                |                |                 |
| Raon-1-mini-preview | pp2048 @ d1024 |  6074.45 ± 19.60 |                |  513.30 ± 1.63 |  503.59 ± 1.63 |   513.90 ± 1.21 |
| Raon-1-mini-preview | tg1024 @ d1024 |     63.47 ± 3.68 |  96.33 ± 24.73 |                |                |                 |
| Raon-1-mini-preview | pp2048 @ d4096 |  6440.00 ± 63.88 |                |  961.83 ± 9.50 |  952.11 ± 9.50 |   961.83 ± 9.50 |
| Raon-1-mini-preview |  tg128 @ d4096 |     64.37 ± 1.74 |   72.33 ± 3.86 |                |                |                 |
| Raon-1-mini-preview | pp2048 @ d4096 |  6503.73 ± 73.95 |                | 952.52 ± 10.77 | 942.81 ± 10.77 |  952.52 ± 10.77 |
| Raon-1-mini-preview | tg1024 @ d4096 |     59.65 ± 4.32 |  95.00 ± 21.95 |                |                |                 |


| model               |                 test |     t/s (total) |         t/s (req) |       peak t/s |   peak t/s (req) |          ttfr (ms) |       est_ppt (ms) |      e2e_ttft (ms) |
|:--------------------|---------------------:|----------------:|------------------:|---------------:|-----------------:|-------------------:|-------------------:|-------------------:|
| Raon-1-mini-preview |  pp2048 @ d1024 (c1) | 5956.26 ± 94.88 |   5956.26 ± 94.88 |                |                  |      523.08 ± 8.27 |      513.71 ± 8.27 |      524.02 ± 8.33 |
| Raon-1-mini-preview |   tg256 @ d1024 (c1) |    64.36 ± 4.72 |      64.36 ± 4.72 |   76.67 ± 7.36 |     76.67 ± 7.36 |                    |                    |                    |
| Raon-1-mini-preview | pp2048 @ d1024 (c10) | 6267.09 ± 16.60 | 1552.77 ± 1187.12 |                |                  |  2867.09 ± 1370.90 |  2857.72 ± 1370.90 |  2867.35 ± 1370.96 |
| Raon-1-mini-preview |  tg256 @ d1024 (c10) |   179.63 ± 3.54 |      22.95 ± 2.45 | 298.33 ± 21.67 |     39.20 ± 3.98 |                    |                    |                    |
| Raon-1-mini-preview | pp2048 @ d1024 (c50) |  5103.87 ± 7.53 |   484.58 ± 707.82 |                |                  | 14042.37 ± 8517.36 | 14033.00 ± 8517.36 | 14042.56 ± 8517.27 |
| Raon-1-mini-preview |  tg256 @ d1024 (c50) |   257.15 ± 1.81 |       8.43 ± 1.70 | 535.33 ± 17.59 |     21.87 ± 6.42 |                    |                    |                    |


| model               |             test |              t/s |       peak t/s |           ttfr (ms) |        est_ppt (ms) |       e2e_ttft (ms) |
|:--------------------|-----------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------------:|
| Raon-1-mini-preview | pp2048 @ d100000 | 1677.51 ± 347.88 |                | 63996.42 ± 15274.56 | 63984.98 ± 15274.56 | 63999.63 ± 15279.07 |
| Raon-1-mini-preview |  tg512 @ d100000 |     54.74 ± 5.47 | 250.00 ± 17.68 |                     |                     |                     |
```