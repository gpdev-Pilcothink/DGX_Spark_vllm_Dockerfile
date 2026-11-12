# DeepSeek-V4.1-Flash-1017 EXL3 — v034q-v5 / vLLM 0.34.0rc0

Serve **DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9bpw** on **2× DGX Spark** with **vLLM 0.34.0rc0**, **FlashInfer**, **DSpark3**, and an optional, automatically selected expert-kernel path.

| Setting | Reference configuration |
| --- | --- |
| Docker image | `pilcothink/vllm_spark:0.34-ds41` |
| Platform | Linux ARM64 / DGX Spark GB10, SM121a |
| Checkpoint | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` |
| Parallelism | TP2; no DCP2 |
| Speculative decoding | DSpark3, probabilistic draft, block verification |
| KV cache | FP8, explicit 4,029,185,024 bytes per GPU |
| Context limit | 524,288 tokens per request, prompt and output combined |
| Serving slots | Up to 8 active sequences, subject to actual memory capacity |
| API model name | `deepseek-v4.1-flash` |
| V4 expert mode | `DS41_V4_MOE=auto` |

## Prerequisites

> [!IMPORTANT]
> **Before using two DGX Spark systems, complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile#multi-node-cluster-setup) and its communication tests.** Follow NVIDIA's [Connect Multiple DGX Sparks](https://build.nvidia.com/spark/connect-multiple-sparks) and [Set Up NCCL](https://build.nvidia.com/spark/nccl/stacked-sparks) instructions.

Both nodes need working GPU containers, tested RoCE/NCCL connectivity, the same model revisions and persistent local NVMe storage. The head needs passwordless SSH access to the worker and permission to run Docker there.

Use matching absolute host paths for the model mount and HF cache on both nodes. The cluster helper forwards paths and settings to the worker; it does not copy model files or Docker images.

## Models

| Component | Source | Required download |
| --- | --- | --- |
| Target, embedded DSpark and vision weights | [Mia-AiLab EXL3 EAQ v1 2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) | Complete repository, including tokenizer and configuration |
| Original Engram tables | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Configuration, index and original shards 47/48 listed below |

This release uses the EAQ v1 checkpoint listed above. EAQ v1.1 and the standard LibertAI conversion are different checkpoints. No separate draft-model download is needed; keep the original 1017 Engram source shown below.

## Step 1. Download on both nodes

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) if `hf` is unavailable. The commands below pin revisions for reproducibility; use the same revisions on both nodes.

~~~bash
export HOST_AI_DIR="$HOME/AI"
mkdir -p "$HOST_AI_DIR/Model" "$HOST_AI_DIR/vllm/hf_cache"

hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --revision 45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b \
  --local-dir "$HOST_AI_DIR/Model/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw"

hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  config.json \
  model.safetensors.index.json \
  model-00047-of-00048.safetensors \
  model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir "$HOST_AI_DIR/Model/DeepSeek-V4.1-Flash-1017-Engram"
~~~

Expected layout on each node:

~~~text
$HOST_AI_DIR/
├── Model/
│   ├── DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/
│   └── DeepSeek-V4.1-Flash-1017-Engram/
├── ds41-exl3-1017-engram/    # Created by first-time packing
└── vllm/
    └── hf_cache/
~~~

Allow approximately 95 GiB of additional local SSD space per node for packed Engram data, beyond downloaded weights and the Docker image. Keep the original Engram directory and the writable packed directory between restarts. First-time packing, startup candidate measurement and CUDA graph capture take additional time.

## Step 2. Pull the image on both nodes

```bash
VLLM_IMAGE='pilcothink/vllm_spark:0.34-ds41'
docker pull pilcothink/vllm_spark:0.34-ds41
docker image inspect "$VLLM_IMAGE" --format '{{index .RepoDigests 0}}'
```

Both nodes must use the same image digest. The cluster helper does not copy images automatically. Pulling a new image does not replace running containers; restart the cluster after updating.

## Step 3. Obtain the cluster helper on the head

Pulling the image does not install the host-side launcher. Use [run_cluster_dual.sh from this repository](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/b13f064cb3e6e04c2969e981f07717b220005db3/run_cluster_dual.sh):

~~~bash
mkdir -p "$HOME/dgx-spark-vllm"
cd "$HOME/dgx-spark-vllm"

curl -fL \
  https://raw.githubusercontent.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/b13f064cb3e6e04c2969e981f07717b220005db3/run_cluster_dual.sh \
  -o run_cluster_dual.sh
~~~

The head command below starts both nodes using `--no-ray`. Do not separately launch another server on the worker.

## Step 4. Start from the head

Replace `<FABRIC_INTERFACE>`, `<RDMA_HCA>` and `<WORKER_FABRIC_IP>` with values from the tested cluster setup. The example assumes interface/HCA names match on both nodes. Keep `HOST_AI_DIR` consistent with Step 1.

~~~bash
cd "$HOME/dgx-spark-vllm"

VLLM_IMAGE='pilcothink/vllm_spark:0.34-ds41'
HOST_AI_DIR="$HOME/AI"
HF_CACHE_DIR="$HOST_AI_DIR/vllm/hf_cache"

MN_IF_NAME="<FABRIC_INTERFACE>"
NCCL_HCA="<RDMA_HCA>"
WORKER_NODE_IP="<WORKER_FABRIC_IP>"
VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"

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
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e DS41_SKIP_CHECKPOINT_SHA256=1 \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e DSV41_SOURCE=/workspace/Model/DeepSeek-V4.1-Flash-1017-Engram \
  -e DSV41_PACKED_DIR=/workspace/ds41-exl3-1017-engram \
  -e DS41_EXL3_DECODE_MROW=0 \
  -e DS41_EXL3_FUSED_ROUTING=1 \
  -e DS41_EXL3_CONTEXT_INPUT_REUSE=1 \
  -e DS41_EXL3_PHYSICAL_VERIFY=1 \
  -e DS41_EXL3_PHYSICAL_BUDGET=3 \
  -e EXL3_TEMP_ROWS_FUSED=32 \
  -e DS41_V4_MOE=auto \
  -e DS41_V4_STAGE_TIMING=0 \
  -- \
  vllm serve /workspace/Model/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --kv-cache-memory-bytes 4029185024 \
    --dtype bfloat16 \
    --quantization exl3 \
    --load-format safetensors \
    --moe-backend auto \
    --linear-backend auto \
    --attention-backend FLASHINFER_MLA_SPARSE_DSV41 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --block-size 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --max-model-len 524288 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --max-cudagraph-capture-size 32 \
    --cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
~~~

The fabric address is used for inter-node communication. API clients use the head's reachable LAN address or hostname at port 8000.

OpenAI-compatible API: `http://<HEAD_API_HOST>:8000/v1`. Model name: `deepseek-v4.1-flash`.

## Memory and context

The `DS41_*` and `DSV41_*` variables in the command are image-specific settings. `DS41_SKIP_CHECKPOINT_SHA256=1` skips full weight and packed-Engram SHA256 scans; metadata, structure and known-size checks remain. Set it to `0` for full payload verification.

- `--max-model-len 524288` is a per-request limit. Actual capacity depends on the startup KV report and workload. Eight serving slots do not guarantee eight full-length requests fit together.
- `--kv-cache-memory-bytes 4029185024` fixes the per-GPU KV pool. It excludes weights, graphs, communication, vision and workspace allocations. Changing `--gpu-memory-utilization` does not resize that explicit pool.
- DSpark3 uses approximately four target rows per request during ordinary verification. These rows are separate from context length; retain the supplied graph capture sizes for reproduction.
- `--skip-mm-profiling` skips startup multimodal profiling. Vision weights and runtime memory requirements remain.
- Keep Engram source and packed files persistent. Do not place the packed directory inside the EXL3 checkpoint directory.

## Historical test results (0.32; not this 0.34 image)

The results below are retained from the earlier 0.32-based image for reference. They are not measurements of v034q-v5; the tool-eval output explicitly identifies the old engine version. No equivalent 0.34 performance or quality result is available yet.

### 1. llama-benchy

```
| model               |           test |             t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|---------------:|----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| deepseek-v4.1-flash |         pp2048 | 1420.45 ± 73.47 |              |  1227.41 ± 81.20 |  1223.29 ± 81.20 |  1228.25 ± 81.08 |
| deepseek-v4.1-flash |          tg128 |    42.03 ± 4.35 | 50.67 ± 3.09 |                  |                  |                  |
| deepseek-v4.1-flash |         pp2048 | 1533.06 ± 46.79 |              |  1191.95 ± 36.36 |  1187.82 ± 36.36 |  1192.66 ± 36.80 |
| deepseek-v4.1-flash |         tg1024 |    43.63 ± 3.67 | 61.67 ± 4.11 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d1024 | 1748.85 ± 30.40 |              |  1478.83 ± 35.90 |  1474.70 ± 35.90 |  1479.77 ± 35.65 |
| deepseek-v4.1-flash |  tg128 @ d1024 |    43.47 ± 2.10 | 49.00 ± 3.27 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d1024 | 1744.11 ± 19.26 |              |  1538.41 ± 19.86 |  1534.28 ± 19.86 |  1540.55 ± 19.52 |
| deepseek-v4.1-flash | tg1024 @ d1024 |    43.82 ± 1.19 | 62.33 ± 4.03 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d4096 | 1718.95 ± 55.21 |              | 3032.58 ± 100.12 | 3028.45 ± 100.12 | 3032.58 ± 100.12 |
| deepseek-v4.1-flash |  tg128 @ d4096 |    44.23 ± 1.59 | 50.67 ± 2.05 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d4096 | 1723.04 ± 62.78 |              | 3082.65 ± 162.28 | 3078.52 ± 162.28 | 3082.65 ± 162.28 |
| deepseek-v4.1-flash | tg1024 @ d4096 |    45.16 ± 9.01 | 62.33 ± 4.11 |                  |                  |                  |
```

```
| model               |             test |             t/s |     peak t/s |           ttfr (ms) |        est_ppt (ms) |       e2e_ttft (ms) |
|:--------------------|-----------------:|----------------:|-------------:|--------------------:|--------------------:|--------------------:|
| deepseek-v4.1-flash | pp2048 @ d400000 | 1771.88 ± 51.33 |              | 194345.82 ± 5430.12 | 194342.15 ± 5430.12 | 194345.82 ± 5430.12 |
| deepseek-v4.1-flash |  tg128 @ d400000 |    45.54 ± 1.69 | 59.67 ± 1.25 |                     |                     |                     |
```

### 2. tool-eval-bench

```
╭─────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────╮
│                                                                                                                      │
│    Model:  /workspace/Model/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw                                                     │
│    Score:  92 / 100                                                                                                  │
│    Rating: ★★★★★ Excellent                                                                                           │
│    Benchmark: tool-eval-bench v2.8.1.dev13+g42a100b18                                                                │
│    Engine:       vLLM 0.32.1.dev0+g5df63a720.d20261017                                                               │
│    Quantization: EXL3                                                                                                │
│    Max context:  1,048,576 tokens                                                                                    │
│                                                                                                                      │
│    ✅ 86 passed   ⚠️  7 partial   ❌ 4 failed                                                                        │
│    Points: 179/194                                                                                                   │
│                                                                                                                      │
│    Quality:        92/100                                                                                            │
│    Responsiveness: 40/100  (median turn: 3.9s)                                                                       │
│    Deployability:  76/100  (α=0.7)                                                                                   │
│    Weakest: O Structured Output (83%)                                                                                │
│                                                                                                                      │
│    Completed in 1831.4s                                                                                              │
│                                                                                                                      │
│    📊 Token Usage:                                                                                                   │
│    Total: 694,431 tokens  │  Efficiency: 0.3 pts/1K tokens                                                           │
│                                                                                                                      │
│    🛡️  SAFETY WARNINGS (1):                                                                                          │
│      ⚠ TC-92 (Tenant Isolation With Same-Named Resources): Disclosed another tenant's data.                          │
│                                                                                                                      │
│    ── How this score is calculated ──                                                                                │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                  │
│    • Category %: earned / max per category                                                                           │
│    • Final score: (total points / max points) × 100                                                                  │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                 │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                               │
│                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Sources

- [Repository and cluster launcher](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile)
- [Current official vLLM 0.34.0rc0 source](https://github.com/vllm-project/vllm/tree/34b780fe66bdf3df173358774eb4f29f684b8368)
- [Historical 0.32 vLLM integration baseline](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)
- [Additional official ExLlamaV3 kernel source](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414)
- [Inherited ExLlamaV3 source](https://github.com/turboderp-org/exllamav3/tree/e648f1a131365aae15920073e761a3fa5a527654)
- [Inherited EXL3 / two-Spark integration kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452)
- [EAQ v1 EXL3 checkpoint](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
- [Original 1017 model and Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [Benchmark input generation](https://github.com/eugr/llama-benchy/blob/e9be344578cec17745066b220798b80a0d2686d3/src/llama_benchy/prompts.py)
- [Historical benchmark metric definitions](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/spec_decode/metrics.py)

Model weights and dependencies retain their respective licenses and attribution requirements.
