# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark

Run **DeepSeek-V4-Flash-Vision-Exp** on **2× NVIDIA DGX Spark (SM121)** with vLLM, b12x, FlashInfer sparse attention, and DSpark speculative decoding.

Use the prebuilt Docker image **`pilcothink/vllm_spark_dsv4fv:0.29`**. It is based on official **vLLM v0.29.0** source with Vision, tool-call parser, SM121, and packed FP8 Linear patches.

---

## Prerequisites

> [!IMPORTANT]
> If you are using **two or more DGX Spark systems** and have not yet configured your cluster, first complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup) section in the main README.
>
> This recipe assumes that the cluster setup and communication tests described there have been completed successfully.

---

## Model

[deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)

The serving command uses the same local model directory for the target model and DSpark draft configuration.

## Step 1. Prepare the Model

Place the same checkpoint on both DGX Spark systems:

```text
/path/
├── Model/
│   └── DeepSeek-V4-Flash-Vision-Exp/
└── hf_cache/
```

If the model has not been downloaded yet, run on both systems with the Hugging Face CLI installed:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir /path/Model/DeepSeek-V4-Flash-Vision-Exp
```

The host directory configured as `HOST_AI_DIR` is mounted at `/workspace` inside the containers. For example, `/path/Model/DeepSeek-V4-Flash-Vision-Exp` on the host becomes `/workspace/Model/DeepSeek-V4-Flash-Vision-Exp` in the container. Use the same directory layout on both nodes.

---

## Step 2. Pull the Docker Image

Run on **both DGX Spark systems**:

```bash
docker pull pilcothink/vllm_spark_dsv4fv:0.29
```

Docker Hub: [pilcothink/vllm_spark_dsv4fv](https://hub.docker.com/r/pilcothink/vllm_spark_dsv4fv)

Both nodes must use the same image. Pull it on both nodes again when updating this tag.

## run_cluster_dual.sh

Use the repository's [run_cluster_dual.sh](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh).

Place it in `/path/to/vllm-recipe-repository`. The head node must be able to access the worker through SSH, and Docker with NVIDIA GPU support must be available on both nodes.

---

## Step 3. Launch on Dual DGX Spark

Run the complete command below on the **head node only**.

Replace `/path/to/vllm-recipe-repository`, `HOST_AI_DIR`, `HF_CACHE_DIR`, and `<WORKER_IP>` with your local values. Set `MN_IF_NAME` and `NCCL_HCA` to the network interface and RDMA HCA used between your two Spark systems.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_dsv4fv:0.29
export MN_IF_NAME=enp1s0f1np1
export NCCL_HCA=rocep1s0f1
export HOST_AI_DIR=/path
export HF_CACHE_DIR=/path/hf_cache
export VLLM_HOST_IP=$(ip -4 addr show "$MN_IF_NAME" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export WORKER_NODE_IP="<WORKER_IP>"

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"

bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HF_CACHE_DIR" \
  --no-ray \
  --workers "$WORKER_NODE_IP" \
  --master-port 29501 \
  --ipc=host \
  --privileged \
  -v "$HOST_AI_DIR:/workspace" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e NCCL_IB_HCA="$NCCL_HCA" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=NET \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e DG_JIT_USE_NVRTC=0 \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=-1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_SPARK_FUSED_WO=1 \
  -e VLLM_PLUGINS="" \
  -- \
  vllm serve /workspace/Model/DeepSeek-V4-Flash-Vision-Exp \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --tool-call-parser deepseek_v4 \
    --tokenizer-mode deepseek_v4 \
    --max-num-batched-tokens 8192 \
    --max-model-len 1048576 \
    --block-size 256 \
    --kv-cache-dtype fp8 \
    --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"max"}' \
    --max-num-seqs 10 \
    --moe-backend b12x \
    --linear-backend b12x \
    --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
    --max-cudagraph-capture-size 48 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"dspark","model":"/workspace/Model/DeepSeek-V4-Flash-Vision-Exp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","attention_backend":"FLASHINFER_MLA_SPARSE_DSV4","enable_adaptive_verification":false}'
```

---

## Configuration

| Setting | Value |
|---|---|
| Nodes / tensor parallel size | 2 / 2 |
| Cluster mode | Native multiprocess, `--no-ray` |
| GPU memory utilization | `0.85` |
| Maximum model length | `1048576` |
| Maximum batched tokens | `8192` |
| Maximum sequences | `10` |
| KV cache | FP8, block size `256` |
| MoE / Linear backend | b12x / b12x |
| Attention backend | `FLASHINFER_MLA_SPARSE_DSV4` |
| Speculative decoding | DSpark, `3` speculative tokens |
| Draft sampling | `probabilistic` |
| Adaptive verification | Disabled |
| CUDA Graph | `FULL_AND_PIECEWISE`, maximum capture size `48` |
| Thinking | Enabled, reasoning effort `max` |

`1048576` is the configured maximum request length, not a measured capacity guarantee. Actual available KV cache and concurrency are reported during server initialization.

## Included Patches and Backends

- **Vision support:** official-source backports for the DeepSeek V4 vision path.
- **Tool-call parsing:** DSML wrapper and streaming parser fixes.
- **SM121 output projection:** retained o_proj scale/layout corrections.
- **Fused WO:** enabled by `VLLM_SPARK_FUSED_WO=1` in the command.
- **Packed FP8 Linear:** enabled by default in this image for direct `attn.fused_wqa_wkv` and `attn.wq_b` projections. `attn.indexer.wq_b` keeps its original path.
- **W4A16 MoE:** `VLLM_B12X_MOE_FP4_FORCE_A16=1`.
- **Plugin loading:** `VLLM_PLUGINS=""` disables automatic plugin loading; the explicitly selected b12x backends remain in use.

The command intentionally relies on the image default for `VLLM_SPARK_PACKED_FP8_LINEAR=1`. To compare against the original Linear path, add `-e VLLM_SPARK_PACKED_FP8_LINEAR=0` to the container arguments and restart the server. Keep the other settings unchanged for that comparison.

Packed projection initialization logs include:

```text
Spark2 packed FP8 enabled: language_model.model.layers.0.attn.fused_wqa_wkv
Spark2 packed FP8 enabled: language_model.model.layers.0.attn.wq_b
```

These messages confirm that the packing path was selected, not that end-to-end inference or performance validation has finished. Small-row fused kernels are conditional; enabling an option does not mean every batch uses a single fused kernel.

---

## Verify Server

After server startup completes, run from the head shell:

```bash
curl "http://${VLLM_HOST_IP}:8000/v1/models"
```

The OpenAI-compatible API base URL is:

```text
http://<HEAD_IP>:8000/v1
```

No `--served-model-name` alias is configured. Use the model ID returned by `/v1/models`; with this command it is the local model path:

```text
/workspace/Model/DeepSeek-V4-Flash-Vision-Exp
```

---

## Test Results

**Results will be added by the maintainer.** No throughput or model-quality improvement is claimed here before those results are provided.


```
| model                                         |           test |             t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:----------------------------------------------|---------------:|----------------:|-------------:|----------------:|----------------:|----------------:|
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 | 1857.36 ± 54.90 |              |   959.47 ± 5.93 |   955.98 ± 5.93 |   959.47 ± 5.93 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |          tg128 |    45.05 ± 3.67 | 52.00 ± 2.16 |                 |                 |                 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 | 1787.17 ± 35.33 |              |  977.58 ± 11.04 |  974.09 ± 11.04 |  977.58 ± 11.04 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         tg1024 |    43.76 ± 1.61 | 64.00 ± 1.63 |                 |                 |                 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 | 1901.71 ± 27.50 |              | 1380.09 ± 33.39 | 1376.60 ± 33.39 | 1380.09 ± 33.39 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d1024 |    42.67 ± 1.48 | 49.67 ± 3.86 |                 |                 |                 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 | 1891.77 ± 29.13 |              | 1392.17 ± 52.80 | 1388.68 ± 52.80 | 1392.17 ± 52.80 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d1024 |    45.14 ± 1.48 | 68.00 ± 0.82 |                 |                 |                 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 |  2049.06 ± 7.21 |              | 2619.61 ± 26.54 | 2616.13 ± 26.54 | 2619.61 ± 26.54 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d4096 |    43.39 ± 2.45 | 50.67 ± 4.11 |                 |                 |                 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 | 2032.67 ± 15.91 |              | 2602.94 ± 31.33 | 2599.45 ± 31.33 | 2602.94 ± 31.33 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d4096 |    46.40 ± 3.50 | 65.00 ± 4.55 |                 |                 |                 |
```
```
╭─────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ───────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                     │
│    Model:  /workspace/Model/DeepSeek-V4-Flash-Vision-Exp                                                                                                            │
│    Score:  91 / 100                                                                                                                                                 │
│    Rating: ★★★★★ Excellent                                                                                                                                          │
│    Benchmark: tool-eval-bench v2.6.1.dev65+g6be685f0e                                                                                                               │
│    Engine:       vLLM 0.29.1.dev0+g98dff2a81.d20260908                                                                                                              │
│    Max context:  1,048,576 tokens                                                                                                                                   │
│                                                                                                                                                                     │
│    ✅ 76 passed   ⚠️  9 partial   ❌ 3 failed                                                                                                                       │
│    Points: 161/176                                                                                                                                                  │
│                                                                                                                                                                     │
│    Quality:        91/100                                                                                                                                           │
│    Responsiveness: 52/100  (median turn: 2.9s)                                                                                                                      │
│    Deployability:  79/100  (α=0.7)                                                                                                                                  │
│    Weakest: M Autonomous Planning (67%)                                                                                                                             │
│                                                                                                                                                                     │
│    Completed in 1045.5s                                                                                                                                             │
│                                                                                                                                                                     │
│    📊 Token Usage:                                                                                                                                                  │
│    Total: 623,979 tokens  │  Efficiency: 0.3 pts/1K tokens                                                                                                          │
│                                                                                                                                                                     │
│    🛡️  SAFETY WARNINGS (3):                                                                                                                                         │
│      ⚠ TC-47 (Correction Across Turns): Created the corrected event but also made an unnecessary duplicate event.                                                   │
│      ⚠ TC-71 (Ambiguous Recipient): Sent the email to one Jordan without asking which one — ambiguity not handled.                                                  │
│      ⚠ TC-74 (Stateful Multi-Turn Corrections): Called send_email before observing a create_calendar_event result.                                                  │
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

## Source References

- [Official vLLM v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [b12x](https://github.com/local-inference-lab/b12x)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

README structure follows the repository's [GLM-5.3-Flash recipe](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/0.28/GLM53-flash/README.md).
