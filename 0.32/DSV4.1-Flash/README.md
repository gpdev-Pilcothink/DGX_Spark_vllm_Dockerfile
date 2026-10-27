# DeepSeek-V4.1-Flash-1017 EXL3 — v4 on 2× DGX Spark

Serve **DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw** on **2× DGX Spark** with **vLLM 0.32**, **FlashInfer**, **DSpark3**, and an optional, automatically selected expert-kernel path.

V4 inherits the cleaned v3 image and adds expert-kernel candidates sourced from official ExLlamaV3 1.6.5, with explicit row-capacity, warp-count and shared-memory edits. The core expert arithmetic is upstream ExLlama code. V4 compares the candidates against the inherited path at startup and reports whether stock, official-reference or modified kernels were selected.

| Setting | Reference configuration |
| --- | --- |
| Docker image | `pilcothink/vllm_ds41:0.32-exl3-v4` |
| Platform | Linux ARM64 / DGX Spark GB10, SM121a |
| Checkpoint | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` |
| Parallelism | TP2; no DCP2 |
| Speculative decoding | DSpark3, probabilistic draft, block verification |
| KV cache | FP8, explicit 3,029,185,024 bytes per GPU |
| Context limit | 524,288 tokens per request, prompt and output combined |
| Serving slots | Up to 8 active sequences, subject to actual memory capacity |
| API model name | `deepseek-v4.1-flash` |
| V4 expert mode | `DS41_V4_MOE=auto` |

## Prerequisites

Complete the repository's [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile#multi-node-cluster-setup), following NVIDIA's [Connect Multiple DGX Sparks](https://build.nvidia.com/spark/connect-multiple-sparks) and [Set Up NCCL](https://build.nvidia.com/spark/nccl/stacked-sparks) instructions.

Both nodes need working GPU containers, tested RoCE/NCCL connectivity, the same model revisions and persistent local NVMe storage. The head needs passwordless SSH access to the worker and permission to run Docker there.

Use matching absolute host paths for the model mount and HF cache on both nodes. The cluster helper forwards paths and settings to the worker; it does not copy model files or Docker images.

## Models

| Component | Source | Required download |
| --- | --- | --- |
| Target, embedded DSpark and vision weights | [LibertAIDAI EXL3 2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | Complete repository, including tokenizer and configuration |
| Original Engram tables | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Configuration, index and original shards 47/48 listed below |

The standard EXL3 checkpoint is the reference here. EAQ v1/v1.1 are separate checkpoint layouts and should not be substituted when reproducing these settings. No separate draft-model download is needed.

## Step 1. Download on both nodes

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) if `hf` is unavailable. The commands below pin revisions for reproducibility; use the same revisions on both nodes.

~~~bash
export HOST_AI_DIR="$HOME/AI"
mkdir -p "$HOST_AI_DIR/Model" "$HOST_AI_DIR/vllm/hf_cache"

hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
  --revision 0be63c3337efbb2a0061d5b8bb4006fa186d433f \
  --local-dir "$HOST_AI_DIR/Model/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"

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
│   ├── DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/
│   └── DeepSeek-V4.1-Flash-1017-Engram/
├── ds41-exl3-1017-engram/    # Created by first-time packing
└── vllm/
    └── hf_cache/
~~~

Allow approximately 95 GiB of additional local SSD space per node for packed Engram data, beyond downloaded weights and the Docker image. Keep the original Engram directory and the writable packed directory between restarts. First-time packing, startup candidate measurement and CUDA graph capture take additional time.

## Step 2. Pull the v4 image on both nodes

~~~bash
docker pull pilcothink/vllm_ds41:0.32-exl3-v4
~~~

Both nodes must use the same built image. Compare the pulled digest on each node:

~~~bash
docker image inspect pilcothink/vllm_ds41:0.32-exl3-v4 \
  --format '{{index .RepoDigests 0}}'
~~~

V4 also compares its native library hash across TP ranks. Two independent compilations can differ because of build IDs and line information, so deploy one built image to both nodes. Pulling a new tag does not replace an already running container; restart the cluster after updating.

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

VLLM_IMAGE=pilcothink/vllm_ds41:0.32-exl3-v4
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
  vllm serve /workspace/Model/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --kv-cache-memory-bytes 3029185024 \
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

## V4 behavior and switches

| Setting | Behavior |
| --- | --- |
| `DS41_V4_MOE=auto` | At startup, screen and time supported expert-kernel candidates against the inherited v3 path, then activate a common passing plan across TP ranks. |
| `DS41_V4_MOE=off` | Use the inherited expert path; skip new-library loading, workspace allocation and candidate selection. |
| `DS41_V4_STAGE_TIMING=0` | Disable the optional per-rank stage observer. Set 1 before restart for bounded stdout timing reports; keep this setting equal across comparisons. |
| `DS41_EXL3_DECODE_MROW=0` | Retain the established MROW-disabled setting. |
| `DS41_EXL3_FUSED_ROUTING=1` | Retain v3's small-batch routing metadata fusion. |
| `DS41_EXL3_CONTEXT_INPUT_REUSE=1` | Retain per-call FP16 input reuse across eligible DSpark context-KV projections. |
| `DS41_EXL3_PHYSICAL_VERIFY=1`, `DS41_EXL3_PHYSICAL_BUDGET=3` | With draft width 3, verify the normal four target rows and bypass the unnecessary full-budget controller. |
| `DSV41_SOURCE` | Container path to the original 1017 Engram source. |
| `DSV41_PACKED_DIR` | Writable, persistent per-node packed Engram directory. |
| `DS41_SKIP_CHECKPOINT_SHA256=1` | Skip full raw-weight and existing packed-Engram payload SHA256 scans; metadata, structure and known-size checks remain. |

The `DS41_*` and `DSV41_*` variables are image-specific integration settings, not official vLLM CLI options. Set `DS41_SKIP_CHECKPOINT_SHA256=0`, or omit it, for full payload verification. Skipping SHA does not skip model reads or first-time packing.

### What is changed in v4

The additional expert kernels come from official ExLlamaV3 1.6.5 at `cdc785db89e263e65e5849cb53e1ada156411414`. Three explicit device-source edits produce the modified candidates:

1. Compile expert-run capacities 1/4/8 instead of reserving 8 for every call.
2. Compile both 8- and 16-warp blocks.
3. Remove the unused K3 staging reservation in the SM121a register-decoding path.

Wide/narrow down-projection tiles are existing upstream options that the image compares. The image-specific native bridge, workspace and startup-selection code connect these candidates to v3. Core MoE arithmetic and trellis decoding remain credited to ExLlamaV3.

Eligible layers require hidden size 5120, an intermediate dimension divisible by 128 up to 4096, top-k 1–6, at most 384 local experts, and uniform integer K2/K3/K4 mul1/non-MCG projections. Selection covers physical rows 1/3/4/8. Other shapes, unsupported packing, unselected calls and large prefill use the inherited path. Eligible K4 draft experts are screened as well as target experts.

### What auto selects

For each packing/shape group, startup uses actual loaded weights with bounded synthetic activations and dispersed/concentrated routing. It screens numerical outputs, then measures alternating CUDA graph replays of the inherited path and the candidates. A candidate must improve measured local operator time by at least 5% on both routing patterns on every TP rank.

Selected candidates are then screened on each supported loaded layer, including graph mutation, invalid routes, zero weights and recovery. Only the common passing layer/row paths activate. The numerical criterion is finite outputs and maximum absolute difference/reference peak <=0.003. This is a bounded numerical screen, not a full model-quality score or bit equality.

The plan is fixed after startup. Normal generation does not rerun tuning or perform tuning collectives. A local operator win does not guarantee a full-serving speedup.

### Check which kernels are active

Look for:

- `DS41_V4_TUNE_BEGIN` and `DS41_V4_TUNE`: startup candidate measurements and group choices.
- `DS41_V4_NATIVE_READY`: counts of selected layer/row paths under `stock`, `official_reference` and `custom`.
- `DS41_PHYSICAL_VERIFY_BYPASS`: expected for fixed 3 / draft 3, which still verifies all three candidates using four target rows.

`official_reference` means unmodified official expert kernels were selected. `custom` means the directly modified variants were selected. Zero custom selections must not be described as a successful custom-kernel acceleration. If no candidate meets the criteria, stock remains selected. Native launch errors stop startup; they are not retried through a second kernel after partial execution.

## API check

~~~bash
curl "http://<HEAD_API_HOST>:8000/v1/models"
~~~

OpenAI-compatible base URL: `http://<HEAD_API_HOST>:8000/v1`  
Model name: `deepseek-v4.1-flash`.

## Memory and context

- `--max-model-len 524288` is a per-request limit. Actual capacity depends on the startup KV report and workload. Eight serving slots do not guarantee eight full-length requests fit together.
- `--kv-cache-memory-bytes 3029185024` fixes the per-GPU KV pool. It excludes weights, graphs, communication, vision and workspace allocations. Changing `--gpu-memory-utilization` does not resize that explicit pool.
- DSpark3 uses approximately four target rows per request during ordinary verification. These rows are separate from context length; retain the supplied graph capture sizes for reproduction.
- `--skip-mm-profiling` skips startup multimodal profiling. Vision weights and runtime memory requirements remain.
- Keep Engram source and packed files persistent. Do not place the packed directory inside the EXL3 checkpoint directory.

## Test

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

## Validation and provenance

Local development checks included 380 numerical/graph-replay cases on RTX 4090 / SM89 synthetic EXL3 tensors, SM121a device-code cross-compilation, and source/installer/CPU control-flow checks. Those device fixtures used official ExLlama 1.6.5 as the reference. They do not certify the deployed image's full-checkpoint quality, every TP2 workload or end-to-end speed.

The official expert-kernel baseline, the three source edits and the selected runtime path are distinct. This image does not claim a new independently invented cooperative MoE algorithm. DeepInfer was reviewed as research; its engine is not included in this image.

The image carries source/native hashes, variant definitions and the applied source diff under `/opt/ds41-native-v4` and `/opt/ds41-decode-v4`. Existing v3 source/native integrity checks and metadata-driven loading are retained. Numerical differences from a changed reduction order can affect close model decisions; the startup tolerance is not a model-quality guarantee.

## Sources

- [Repository and cluster launcher](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile)
- [Pinned vLLM source](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)
- [Additional official ExLlamaV3 kernel source](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414)
- [Inherited ExLlamaV3 source](https://github.com/turboderp-org/exllamav3/tree/e648f1a131365aae15920073e761a3fa5a527654)
- [Inherited EXL3 / two-Spark integration kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452)
- [Standard LibertAI EXL3 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
- [Original 1017 model and Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [Benchmark input generation](https://github.com/eugr/llama-benchy/blob/e9be344578cec17745066b220798b80a0d2686d3/src/llama_benchy/prompts.py)
- [vLLM speculative metric definitions](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/spec_decode/metrics.py)

Model weights and dependencies retain their respective licenses and attribution requirements.
