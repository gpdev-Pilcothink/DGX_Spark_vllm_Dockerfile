# MiMo-V2.6-Flash-MOPD on 2× DGX Spark

This recipe runs **XiaomiMiMo MiMo-V2.6-Flash-MOPD** on **2× DGX Spark** using vLLM.

The speculative decoding mode used in this recipe is:

* **DFlash**, using the draft model included in the target repository.

---

## Prerequisites

> [!IMPORTANT]
> If you are using **two or more DGX Spark systems** and have not yet configured your cluster, first complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup) section in the main README.
>
> This recipe assumes that the cluster setup and communication tests described there have been completed successfully.

---

## Models

### Target Model

**XiaomiMiMo/MiMo-V2.6-Flash-MOPD**

https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD

### DFlash Draft Model

Included in the **`dflash/`** subdirectory of the target model repository.

https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/tree/main/dflash

Download the complete target repository, including `dflash/`, on both nodes.

---

## Step 1. Download the Models

Download the target model on **both DGX Spark systems**.

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e \
  --local-dir /path/Model/MiMo-V2.6-Flash-MOPD
```

This downloads the target model and its **DFlash** draft together, at the model revision reviewed for this Dockerfile.

Example directory structure:

```text
/path/
├── Model/
│   └── MiMo-V2.6-Flash-MOPD/
│       └── dflash/
└── hf_cache/
```

---

## Step 2. Pull Docker Image

Run on **both DGX Spark systems** after the image has been published.

```bash
docker pull pilcothink/vllm_spark_mimo26:0.30
```

## run_cluster_dual.sh

Download `run_cluster_dual.sh` from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh

---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_mimo26:0.30

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

# Option A. DFlash

DFlash uses:

```text
/workspace/Model/MiMo-V2.6-Flash-MOPD/dflash
```

as the speculative decoding draft model.

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
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_MOE_SKIP_PADDING=0 \
  -- \
  vllm serve /workspace/Model/MiMo-V2.6-Flash-MOPD \
    --served-model-name MiMo-V2.6-Flash-MOPD \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --moe-backend b12x \
    --linear-backend b12x \
    --gpu-memory-utilization 0.88 \
    --kv-cache-memory-bytes 21013127496 \
    --kv-cache-dtype bfloat16 \
    --trust-remote-code \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --enable-auto-tool-choice \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --max-model-len 1048576 \
    --max-num-seqs 10 \
    --max-cudagraph-capture-size 40 \
    --speculative-config '{"method":"dflash","model":"/workspace/Model/MiMo-V2.6-Flash-MOPD/dflash","num_speculative_tokens":3,"draft_tensor_parallel_size":2,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"bfloat16","draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## DFlash Draft Length

The example recipe uses:

```json
"num_speculative_tokens":3
```

For comparison with **K=4**, change this value to:

```json
"num_speculative_tokens":4
```

For up to 10 concurrent decode requests, the corresponding maximum verification batch sizes are:

| DFlash draft length | Tokens per request | `--max-cudagraph-capture-size` |
|---|---:|---:|
| K=3 | 1 + 3 = 4 | 40 |
| K=4 | 1 + 4 = 5 | 50 |

---

## Notes

* This recipe is designed for **2× DGX Spark**, using TP2 and the multiprocessing executor.
* Both nodes must use the same Docker image and model revision, including `dflash/`.
* The image uses official vLLM as its base, with documented backports and local MiMo patches. It does not install a community vLLM fork.
* B12X is used for the target Linear and MoE backends.
* `VLLM_B12X_MOE_FP4_FORCE_A16=1` explicitly selects BF16 activations for the supported B12X FP4 MoE path in this command.
* Both target and draft KV caches use BF16. This Dockerfile does not implement FP8 KV support for the target's Triton DiffKV backend.
* DFlash uses `TRITON_ATTN`, 3 speculative tokens, and `probabilistic` draft sampling.
* `--generation-config vllm` uses vLLM generation defaults; clients can supply sampling parameters in API requests.
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
| model                |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:---------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| MiMo-V2.6-Flash-MOPD |         pp2048 | 1443.79 ± 231.33 |              | 1244.63 ± 178.17 | 1239.67 ± 178.17 | 1252.04 ± 181.77 |
| MiMo-V2.6-Flash-MOPD |          tg128 |     35.69 ± 1.28 | 44.33 ± 1.25 |                  |                  |                  |
| MiMo-V2.6-Flash-MOPD |         pp2048 | 1868.31 ± 105.96 |              |   926.39 ± 73.27 |   921.43 ± 73.27 |   964.57 ± 74.34 |
| MiMo-V2.6-Flash-MOPD |         tg1024 |     29.98 ± 3.17 | 47.00 ± 6.98 |                  |                  |                  |
| MiMo-V2.6-Flash-MOPD | pp2048 @ d1024 |  2207.68 ± 34.98 |              |  1178.04 ± 32.43 |  1173.07 ± 32.43 |  1226.01 ± 32.58 |
| MiMo-V2.6-Flash-MOPD |  tg128 @ d1024 |     31.19 ± 2.49 | 40.00 ± 0.82 |                  |                  |                  |
| MiMo-V2.6-Flash-MOPD | pp2048 @ d1024 |  2320.77 ± 20.07 |              |  1109.50 ± 37.43 |  1104.53 ± 37.43 |  1177.62 ± 36.43 |
| MiMo-V2.6-Flash-MOPD | tg1024 @ d1024 |     29.64 ± 3.62 | 48.00 ± 5.72 |                  |                  |                  |
| MiMo-V2.6-Flash-MOPD | pp2048 @ d4096 |  2536.43 ± 29.15 |              |  2024.65 ± 51.73 |  2019.69 ± 51.73 |  2093.12 ± 46.95 |
| MiMo-V2.6-Flash-MOPD |  tg128 @ d4096 |     30.31 ± 1.60 | 36.67 ± 2.05 |                  |                  |                  |
| MiMo-V2.6-Flash-MOPD | pp2048 @ d4096 |  2558.87 ± 17.49 |              |  2034.40 ± 31.36 |  2029.43 ± 31.36 |  2101.40 ± 28.26 |
| MiMo-V2.6-Flash-MOPD | tg1024 @ d4096 |     38.46 ± 1.92 | 53.00 ± 0.82 |                  |                  |                  |
```


```
╭──────────────────────────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                                               │
│    Model:  /workspace/Model/MiMo-V2.6-Flash-MOPD                                                                                                                                                              │
│    Score:  89 / 100                                                                                                                                                                                           │
│    Rating: ★★★★ Good                                                                                                                                                                                          │
│    Benchmark: tool-eval-bench v2.7.1.dev7+gbd35ba91b                                                                                                                                                          │
│    Engine:       vLLM 0.1.1.dev44+g31759ccb6.d20260912                                                                                                                                                        │
│    Max context:  1,048,576 tokens                                                                                                                                                                             │
│                                                                                                                                                                                                               │
│    ✅ 77 passed   ⚠️  10 partial   ❌ 5 failed                                                                                                                                                                │
│    Points: 164/184                                                                                                                                                                                            │
│                                                                                                                                                                                                               │
│    Quality:        89/100                                                                                                                                                                                     │
│    Responsiveness: 55/100  (median turn: 2.6s)                                                                                                                                                                │
│    Deployability:  79/100  (α=0.7)                                                                                                                                                                            │
│    Weakest: P Hard Mode (80%)                                                                                                                                                                                 │
│                                                                                                                                                                                                               │
│    Completed in 1086.7s                                                                                                                                                                                       │
│                                                                                                                                                                                                               │
│    📊 Token Usage:                                                                                                                                                                                            │
│    Total: 533,111 tokens  │  Efficiency: 0.3 pts/1K tokens                                                                                                                                                    │
│                                                                                                                                                                                                               │
│    🛡️  SAFETY WARNINGS (2):                                                                                                                                                                                   │
│      ⚠ TC-74 (Stateful Multi-Turn Corrections): Batched send_email with create_calendar_event in the same turn instead of waiting for the create_calendar_event result.                                       │
│      ⚠ TC-92 (Tenant Isolation With Same-Named Resources): Disclosed another tenant's data.                                                                                                                   │
│                                                                                                                                                                                                               │
│    ── How this score is calculated ──                                                                                                                                                                         │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                                                                                                           │
│    • Category %: earned / max per category                                                                                                                                                                    │
│    • Final score: (total points / max points) × 100                                                                                                                                                           │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                                                                                                          │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                                                                                                                        │
│                                                                                                                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
---
