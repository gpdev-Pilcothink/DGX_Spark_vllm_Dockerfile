# GLM-5.3-Flash NVFP4 on 2× DGX Spark

This recipe runs **NVIDIA GLM-5.3-Flash-NVFP4** on **2× DGX Spark** using vLLM.

Two speculative decoding modes are available:

* **MTP**
* **DFlash2**

---

## Models

### Target Model

**nvidia/GLM-5.3-Flash-NVFP4**

https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4

### DFlash2 Draft Model

Required only when using DFlash2.

**incoai/GLM-5.3-Flash-DFlash2**

https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2

---

## Step 1. Download the Models

Download the target model on **both DGX Spark systems**.

```bash
hf download nvidia/GLM-5.3-Flash-NVFP4 \
  --local-dir /path/Model/GLM-5.3-Flash-NVFP4
```

If you want to use **DFlash2**, also download:

```bash
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir /path/Model/GLM-5.3-Flash-DFlash2
```

Example directory structure:

```text
/path/
├── Model/
│   ├── GLM-5.3-Flash-NVFP4/
│   └── GLM-5.3-Flash-DFlash2/
└── hf_cache/
```

---

## Step 2. Pull Docker Image

Run on **both DGX Spark systems**.

```bash
docker pull pilcothink/vllm_spark_glm53:0.28
```


## run_cluster_dual.sh

Download `run_cluster_dual.sh` from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh


---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_glm53:0.28

export MN_IF_NAME=enp1s0f1np1
export NCCL_HCA=rocep1s0f1
export WORKER_NODE_IP=<WORKER_IP>

export HOST_AI_DIR=/path
export HF_CACHE_DIR=/path/hf_cache

export VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"
```

Replace the paths, network interface, HCA, and worker IP with values from your own environment.

---

# Option A. MTP

MTP uses the speculative decoding layers included with the target model.

Run on the **head DGX Spark only**.

```bash
bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HF_CACHE_DIR" \
  --no-ray \
  --workers "$WORKER_NODE_IP" \
  --master-port 29501 \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOST_AI_DIR:/workspace" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e NCCL_IB_HCA="$NCCL_HCA" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -- \
  vllm serve /workspace/Model/GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --moe-backend b12x \
    --linear-backend b12x \
    --gpu-memory-utilization 0.9 \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-cudagraph-capture-size 64 \
    --max-num-seqs 10 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"auto"}'
```

---

# Option B. DFlash2

DFlash2 uses:

```text
incoai/GLM-5.3-Flash-DFlash2
```

as an external speculative decoding draft model.

Run on the **head DGX Spark only**.

```bash
bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HF_CACHE_DIR" \
  --no-ray \
  --workers "$WORKER_NODE_IP" \
  --master-port 29501 \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOST_AI_DIR:/workspace" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e NCCL_IB_HCA="$NCCL_HCA" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -- \
  vllm serve /workspace/Model/GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --moe-backend b12x \
    --linear-backend b12x \
    --gpu-memory-utilization 0.9 \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-cudagraph-capture-size 64 \
    --max-num-seqs 10 \
    --speculative-config '{"method":"dflash","model":"/workspace/Model/GLM-5.3-Flash-DFlash2","num_speculative_tokens":5,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"disable_eagle_block_drop":false}'
```

## DFlash2 Draft Length

This Docker image supports the following DFlash2 draft lengths:

```text
num_speculative_tokens = 4
num_speculative_tokens = 5
num_speculative_tokens = 6
num_speculative_tokens = 7
```

For example:

### K=4

```json
"num_speculative_tokens":4
```

### K=5

```json
"num_speculative_tokens":5
```

### K=6

```json
"num_speculative_tokens":6
```

### K=7

```json
"num_speculative_tokens":7
```

The example recipe uses:

```json
"num_speculative_tokens":5
```

Only change `num_speculative_tokens` in the DFlash2 speculative configuration when testing different draft lengths.

---

### Optional: 1M Context Length

1M context serving is also supported on **2× DGX Spark** by manually allocating a 9 GiB KV cache.

Replace the default context settings with:

```bash
--gpu-memory-utilization 0.9 \
--kv-cache-memory=9663676416 \
--max-model-len 1000000 \
```

`--kv-cache-memory=9663676416` allocates **9.0 GiB** of KV cache per worker.

> **Important:** For 1M context serving, **both DGX Spark systems should have at least approximately 118 GiB of available memory before starting the server.**
>
> Check the available memory on both nodes with:
>
> ```bash
> free -h
> ```
>
> Make sure the `available` column reports at least **118 GiB** on each DGX Spark.

The default recipe uses `262144` tokens for a larger memory margin and higher concurrency.

---

## Notes

* This recipe is designed for **2× DGX Spark**.
* Both nodes must use the same Docker image.
* Both nodes must have the same target model.
* DFlash2 additionally requires the same draft model on both nodes.
* The GLM-5.3 target model remains multimodal.
* Do not use `--language-model-only`.
* Do not force `VLLM_KV_CACHE_LAYOUT=BLHNC`.
* B12X is used for the target Linear and MoE backends.
* The target KV cache uses FP8.
* DFlash2 uses `TRITON_ATTN` for the draft attention backend.
* DFlash2 supports `num_speculative_tokens` values from **4 to 7** in this Docker image.
* `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, worker IP, and local paths must be adjusted for your environment.

---

## Verify Server

```bash
curl http://<HEAD_IP>:8000/v1/models
```

The OpenAI-compatible API is available at:

```text
http://<HEAD_IP>:8000
```


---

## ===========Test===========

```
--gpu-memory-utilization 0.9 

==> (EngineCore pid=279) INFO 09-12 07:07:14 [kv_cache_utils.py:2312] GPU KV cache size: 743,165 tokens, Maximum concurrency for 262,144 tokens per request: 2.83x
```


**MTP = 5**

```
| model                                |           test |              t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:-------------------------------------|---------------:|-----------------:|-------------:|------------------:|------------------:|------------------:|
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1204.73 ± 64.41 |              |   1423.06 ± 70.74 |   1418.73 ± 70.74 |   1423.06 ± 70.74 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |          tg128 |     26.81 ± 0.87 | 35.67 ± 2.05 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1314.93 ± 39.89 |              |    1305.57 ± 6.50 |    1301.23 ± 6.50 |    1305.57 ± 6.50 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         tg1024 |     23.68 ± 3.45 | 45.00 ± 2.83 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 | 1534.44 ± 295.32 |              |  1880.93 ± 355.18 |  1876.59 ± 355.18 |  1880.93 ± 355.18 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d1024 |     22.42 ± 0.33 | 30.67 ± 1.70 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  1542.21 ± 36.05 |              |   1676.95 ± 26.13 |   1672.61 ± 26.13 |   1676.95 ± 26.13 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d1024 |     22.99 ± 4.45 | 41.67 ± 8.26 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1555.07 ± 269.50 |              |  3396.51 ± 716.78 |  3392.18 ± 716.78 |  3396.51 ± 716.78 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d4096 |     24.34 ± 2.08 | 35.00 ± 3.74 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1208.21 ± 254.17 |              | 4569.23 ± 1025.35 | 4564.89 ± 1025.35 | 4569.23 ± 1025.35 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d4096 |     23.51 ± 1.70 | 42.33 ± 3.68 |                   |                   |                   |

llama-benchy (0.3.9.dev9+g446dd42fd)
```



**Dflash speculative_tokens = 5**

```
| model                                |           test |              t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:-------------------------------------|---------------:|-----------------:|-------------:|------------------:|------------------:|------------------:|
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  921.46 ± 118.42 |              |  1941.02 ± 319.34 |  1937.29 ± 319.34 |  1941.02 ± 319.34 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |          tg128 |     35.74 ± 2.01 | 48.33 ± 2.49 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1058.97 ± 59.39 |              |   1631.32 ± 89.63 |   1627.59 ± 89.63 |   1631.32 ± 89.63 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         tg1024 |     28.00 ± 3.02 | 57.67 ± 3.30 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |    320.66 ± 7.40 |              |  8130.20 ± 165.87 |  8126.48 ± 165.87 |  8130.20 ± 165.87 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d1024 |     33.79 ± 3.77 | 43.00 ± 4.55 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  855.27 ± 381.10 |              | 4177.52 ± 2548.24 | 4173.80 ± 2548.24 | 4177.52 ± 2548.24 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d1024 |     30.59 ± 4.61 | 53.67 ± 5.31 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 |  1429.45 ± 17.64 |              |   3609.23 ± 38.00 |   3605.51 ± 38.00 |   3632.55 ± 63.88 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d4096 |     33.39 ± 2.06 | 40.00 ± 3.56 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1457.16 ± 249.48 |              |  3792.10 ± 287.95 |  3788.38 ± 287.95 |  3792.10 ± 287.95 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d4096 |     30.53 ± 1.00 | 53.67 ± 5.44 |                   |                   |                   |

llama-benchy (0.3.9.dev9+g446dd42fd)
```


**Dflash speculative_tokens = 7 (official set)**

```
| model                                |           test |              t/s |     peak t/s |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:-------------------------------------|---------------:|-----------------:|-------------:|------------------:|------------------:|------------------:|
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  877.48 ± 122.55 |              |  2014.67 ± 285.73 |  2008.84 ± 285.73 |  2014.67 ± 285.73 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |          tg128 |     39.11 ± 2.19 | 46.33 ± 4.11 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1064.08 ± 21.37 |              |   1660.66 ± 23.22 |   1654.83 ± 23.22 |   1660.66 ± 23.22 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         tg1024 |     30.77 ± 1.23 | 57.33 ± 4.64 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  765.33 ± 332.39 |              | 4511.10 ± 2621.62 | 4505.27 ± 2621.62 | 4511.10 ± 2621.62 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d1024 |     29.07 ± 2.08 | 41.67 ± 3.86 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  580.78 ± 355.86 |              | 6630.03 ± 3049.41 | 6624.20 ± 3049.41 | 6630.03 ± 3049.41 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d1024 |     31.04 ± 3.02 | 64.33 ± 2.62 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1334.81 ± 123.93 |              |  3942.27 ± 457.97 |  3936.44 ± 457.97 |  3942.27 ± 457.97 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d4096 |     29.82 ± 2.67 | 42.00 ± 2.94 |                   |                   |                   |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1338.17 ± 118.41 |              |  3869.36 ± 327.64 |  3863.53 ± 327.64 |  3869.36 ± 327.64 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d4096 |     31.40 ± 2.15 | 63.67 ± 7.59 |                   |                   |                   |

llama-benchy (0.3.9.dev9+g446dd42fd)
```



**tool-eval-bench v2.6.1.dev65+g6be685f0e**

```
╭─────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ───────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                     │
│    Model:  /workspace/Model/GLM-5.3-Flash-NVFP4                                                                                                                     │
│    Score:  94 / 100                                                                                                                                                 │
│    Rating: ★★★★★ Excellent                                                                                                                                          │
│    Benchmark: tool-eval-bench v2.6.1.dev65+g6be685f0e                                                                                                               │
│    Engine:       vLLM 0.28.1rc1.dev475+g6fbb00b18.d20260907                                                                                                         │
│    Max context:  262,144 tokens                                                                                                                                     │
│                                                                                                                                                                     │
│    ✅ 82 passed   ⚠️  2 partial   ❌ 4 failed                                                                                                                       │
│    Points: 166/176                                                                                                                                                  │
│                                                                                                                                                                     │
│    Quality:        94/100                                                                                                                                           │
│    Responsiveness: 31/100  (median turn: 5.2s)                                                                                                                      │
│    Deployability:  75/100  (α=0.7)                                                                                                                                  │
│    Weakest: M Autonomous Planning (50%)                                                                                                                             │
│                                                                                                                                                                     │
│    Completed in 2149.7s                                                                                                                                             │
│                                                                                                                                                                     │
│    📊 Token Usage:                                                                                                                                                  │
│    Total: 577,510 tokens  │  Efficiency: 0.3 pts/1K tokens                                                                                                          │
│                                                                                                                                                                     │
│    🛡️  SAFETY WARNINGS (1):                                                                                                                                         │
│      ⚠ TC-51 (Goal-Level Planning): Called send_email before observing a create_calendar_event result.                                                              │
│                                                                                                                                                                     │
│    ── How this score is calculated ──                                                                                                                               │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                                                                 │
│    • Category %: earned / max per category                                                                                                                          │
│    • Final score: (total points / max points) × 100                                                                                                                 │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                                                                │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                                                                              │
│                                                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
---