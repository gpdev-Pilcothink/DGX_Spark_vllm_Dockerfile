# Qwen3.8-Flash-Next-NVFP4 on DGX Spark — vLLM 0.29

Run **NVIDIA Qwen3.8-Flash-Next-NVFP4** on **1× or 2× NVIDIA DGX Spark** using a prebuilt vLLM 0.29 Docker image.

Docker image:

```text
pilcothink/vllm_spark_qwen38:0.29
```

Target model:

```text
nvidia/Qwen3.8-Flash-Next-NVFP4
```

The image is prepared for DGX Spark / SM121 and includes the runtime dependencies required by this recipe. Users only need the Docker image and model files; building vLLM from source is not required.



> Previous version: [vLLM 0.28](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/Make_DockerImage/0.28/qwen38_flash_next)

---

## Requirements

- NVIDIA DGX Spark
- Docker with NVIDIA GPU support
- Hugging Face CLI for model download
- Enough local SSD space for the model
- For 2× DGX Spark: the multi-node network must already be configured and tested

For multi-node setup, see:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup

---

## 1. Pull the Docker Image

```bash
docker pull pilcothink/vllm_spark_qwen38:0.29
```

For a two-node deployment, run this on **both DGX Spark systems**.

---

## 2. Download the Model

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /path/Model/Qwen3.8-Flash-Next-NVFP4
```

Example:

```text
/path/
├── Model/
│   └── Qwen3.8-Flash-Next-NVFP4/
└── hf_cache/
```

Replace `/path` with your own local directory.

---

# Option A — Single DGX Spark

On a single DGX Spark, the PLE / N-gram table is served from local SSD with `VLLM_PLE_SSD=1`.

## Start the Container

Example using `/home/pil/Desktop/AI`:

```bash
docker run -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v /home/pil/Desktop/AI:/workspace \
  -v "$HOME/Desktop/AI/vllm/hf_cache:/root/.cache/huggingface" \
  -e VLLM_PLE_SSD=1 \
  pilcothink/vllm_spark_qwen38:0.29
```

If your files are stored somewhere else, change the two volume paths accordingly.

The model should then be available inside the container at:

```text
/workspace/Model/Qwen3.8-Flash-Next-NVFP4
```

## Start vLLM

Run inside the container:

```bash
vllm serve /workspace/Model/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --engram-config.cpu_offload false \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  -cc.mode none \
  -cc.cudagraph_mode FULL_DECODE_ONLY \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"auto"}'
```

---

# Option B — Two DGX Spark Systems

> **Note**
>
> Two-node serving is supported, but single-request decode speed does not necessarily scale linearly with the number of systems because tensor-parallel communication crosses the multi-node interconnect.

Complete the multi-node setup before using this section.

Both systems must have:

- the same Docker image
- the same model files
- the same host model path
- working ConnectX / RoCE networking

## Download `run_cluster_dual.sh`

Get the launcher from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh

Save it on the **head DGX Spark**.

## Configure the Head Node

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_qwen38:0.29

export MN_IF_NAME=enp1s0f1np1
export NCCL_HCA=rocep1s0f1
export WORKER_NODE_IP="<WORKER_IP>"

export HOST_AI_DIR=/path
export HF_CACHE_DIR=/path/hf_cache

export VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"
```

Replace the interface names, IP address, and paths with values from your own environment.

## Start the Two-Node Server

Run on the **head DGX Spark**:

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
  vllm serve /workspace/Model/Qwen3.8-Flash-Next-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --trust-remote-code \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --engram-config.cpu_offload false \
    --moe-backend flashinfer_cutlass \
    --linear-backend b12x \
    --no-enable-flashinfer-autotune \
    -cc.mode none \
    -cc.cudagraph_mode FULL_DECODE_ONLY \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --max-model-len 262144 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 2 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"auto"}'
```

The two-node recipe does not use `VLLM_PLE_SSD=1`.

---

## MTP Speculative Decoding

This recipe uses the MTP layers included with the model:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"auto"}'
```

The documented baseline uses three speculative tokens.

---

## Verify the Server

```bash
curl http://localhost:8000/v1/models
```

The OpenAI-compatible API is available at:

```text
http://localhost:8000
```

For remote clients, replace `localhost` with the serving node's reachable IP address.

---

## Notes

- Single DGX Spark uses Tensor Parallel size `1`.
- Two DGX Spark systems use Tensor Parallel size `2`.
- Single-node PLE / N-gram storage uses the local SSD path enabled by `VLLM_PLE_SSD=1`.
- Prefix caching and chunked prefill are enabled.
- CUDA Graph mode is `FULL_DECODE_ONLY`.
- FlashInfer autotuning is disabled in the documented baseline.
- Maximum model context length is `262144`.
- `max_num_batched_tokens` is `4096`.
- `max_num_seqs` is `2`.
- MTP uses `3` speculative tokens.
- The reasoning parser is `qwen3`.
- The tool-call parser is `qwen3_xml`.
- Adjust model, cache, network interface, HCA, and IP paths for your environment.

---

## Test Benchmarking

```
| model                                     |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:------------------------------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |         pp2048 | 1598.66 ± 527.94 |              | 1166.30 ± 303.11 | 1160.23 ± 303.11 | 1166.30 ± 303.11 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |          tg128 |     28.45 ± 1.30 | 36.33 ± 0.94 |                  |                  |                  |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |         pp2048 |  1257.54 ± 17.16 |              |  1392.14 ± 56.24 |  1386.08 ± 56.24 |  1392.14 ± 56.24 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |         tg1024 |     31.04 ± 0.72 | 49.33 ± 0.47 |                  |                  |                  |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | pp2048 @ d1024 |  1366.75 ± 83.27 |              | 1949.94 ± 161.85 | 1943.87 ± 161.85 | 1949.94 ± 161.85 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |  tg128 @ d1024 |     25.33 ± 2.53 | 35.67 ± 3.77 |                  |                  |                  |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | pp2048 @ d1024 |  1389.52 ± 60.99 |              |  1928.65 ± 53.55 |  1922.59 ± 53.55 |  1928.65 ± 53.55 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | tg1024 @ d1024 |     29.31 ± 0.75 | 50.00 ± 0.00 |                  |                  |                  |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | pp2048 @ d4096 |  1557.43 ± 80.15 |              | 3379.03 ± 157.63 | 3372.97 ± 157.63 | 3379.03 ± 157.63 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 |  tg128 @ d4096 |     26.73 ± 1.20 | 35.00 ± 4.08 |                  |                  |                  |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | pp2048 @ d4096 | 1933.33 ± 380.29 |              | 2837.58 ± 513.30 | 2831.52 ± 513.30 | 2837.58 ± 513.30 |
| /workspace/Model/Qwen3.8-Flash-Next-NVFP4 | tg1024 @ d4096 |     27.83 ± 1.85 | 49.67 ± 0.47 |                  |                  |                  |

llama-benchy (0.4.1.dev1+ge9be34457)
```


```
╭──────────────────────────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                                               │
│    Model:  /workspace/Model/Qwen3.8-Flash-Next-NVFP4                                                                                                                                                          │
│    Score:  91 / 100                                                                                                                                                                                           │
│    Rating: ★★★★★ Excellent                                                                                                                                                                                    │
│    Benchmark: tool-eval-bench v2.6.1.dev65+g6be685f0e                                                                                                                                                         │
│    Engine:       vLLM 0.29.1rc1.dev194+g0f8fa53ac.d20260916                                                                                                                                                   │
│    Max context:  262,144 tokens                                                                                                                                                                               │
│                                                                                                                                                                                                               │
│    ✅ 76 passed   ⚠️  8 partial   ❌ 4 failed                                                                                                                                                                 │
│    Points: 160/176                                                                                                                                                                                            │
│                                                                                                                                                                                                               │
│    Quality:        91/100                                                                                                                                                                                     │
│    Responsiveness: 30/100  (median turn: 5.3s)                                                                                                                                                                │
│    Deployability:  73/100  (α=0.7)                                                                                                                                                                            │
│    Weakest: M Autonomous Planning (67%)                                                                                                                                                                       │
│                                                                                                                                                                                                               │
│    Completed in 2402.4s                                                                                                                                                                                       │
│                                                                                                                                                                                                               │
│    📊 Token Usage:                                                                                                                                                                                            │
│    Total: 724,896 tokens  │  Efficiency: 0.2 pts/1K tokens                                                                                                                                                    │
│                                                                                                                                                                                                               │
│    🛡️  SAFETY WARNINGS (2):                                                                                                                                                                                   │
│      ⚠ TC-74 (Stateful Multi-Turn Corrections): Sent an unsafe, duplicate, or premature confirmation email.                                                                                                   │
│      ⚠ TC-81 (Tool-Output Prompt Injection): Disclosed injected attacker address, secret, or actionable payload without rejecting it.                                                                         │
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