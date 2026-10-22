# DeepSeek-V4.1-Flash-1017 EXL3 on 2× DGX Spark

This recipe runs **DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw** on **2× DGX Spark** using **vLLM 0.32**, **FlashInfer**, and **ExLlamaV3**.

The configuration uses **TP2**, **DSpark with 3 speculative tokens**, and a **1,048,576-token context limit**. EXL3 weights reside in unified memory across both nodes. The original **Engram N-gram tables remain on local NVMe SSDs** and are read as needed.

---

## Prerequisites

> [!IMPORTANT]
> Before using two DGX Spark systems, complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup) section in the main README.
>
> This recipe assumes that the cluster setup and communication tests described there have completed successfully.

Both systems need local NVMe storage for the same EXL3 checkpoint, the original Engram source files, and their own packed Engram files.

---

## Models

### Target Model

[**LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw**](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)

Includes the target model, tokenizer, vision components and DSpark weights. No separate draft model download is required.

### Original Engram Tables

[**deepseek-ai/DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

Only the configuration, weight index and two Engram shards listed below are required.

---

## Step 1. Download the Models

Prepare the following pinned model files on **each node's local NVMe**.

**1. EXL3 model — complete checkpoint, including tokenizer, vision and DSpark weights**

```bash
hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
  --revision 0be63c3337efbb2a0061d5b8bb4006fa186d433f \
  --local-dir "$HOME/Desktop/AI/Model/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
```

**2. Original 1017 Engram — only these four files**

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  config.json \
  model.safetensors.index.json \
  model-00047-of-00048.safetensors \
  model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir "$HOME/Desktop/AI/Model/DeepSeek-V4.1-Flash-1017-Engram"
```

The EXL3 checkpoint does not include the original N-gram tables. The two original shards supply those tables; the remaining original model shards are not required by this recipe.

Example directory structure on each node:

```text
~/Desktop/AI/
├── Model/
│   ├── DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/
│   └── DeepSeek-V4.1-Flash-1017-Engram/
├── ds41-exl3-1017-engram/    # Created during first-time packing
└── vllm/
    └── hf_cache/
```

> [!IMPORTANT]
> First startup packs each node's own Engram rows. Allow approximately **95 GiB of additional local SSD space per node**, beyond the downloaded checkpoints and Docker image.
>
> The packed directory must be writable and persistent. Keep both the original Engram files and the packed directory between restarts. First-time packing and kernel initialization can take longer than subsequent startups.

---

## Step 2. Pull Docker Image

Run on **both DGX Spark systems**:

```bash
docker pull pilcothink/vllm_ds41:0.32-exl3
```

Both nodes must use the **same image digest**.

### run_cluster_dual.sh

Use [run_cluster_dual.sh from this repository](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh). The following example assumes it is located in `~/Desktop/storage/DGX_Spark/vllm`.

---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**:

```bash
cd ~/Desktop/storage/DGX_Spark/vllm

export VLLM_IMAGE=pilcothink/vllm_ds41:0.32-exl3

export MN_IF_NAME=enp1s0f1np1
export NCCL_HCA=rocep1s0f1
export WORKER_NODE_IP=169.254.139.122

export HOST_AI_DIR="$HOME/Desktop/AI"
export HF_CACHE_DIR="$HOME/Desktop/AI/vllm/hf_cache"

export VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"
```

Replace the interface, HCA, worker IP and paths with values for your environment. The mounted host directory must exist on **both nodes** and provide the same `/workspace/Model/...` paths inside their containers.

---

## Step 4. Start the Server

Run on the **head DGX Spark only**, in the same shell as Step 3.

| Setting | Value |
| --- | --- |
| Tensor parallelism | 2 |
| Speculative decoding | DSpark, 3 speculative tokens |
| Maximum context | 1,048,576 tokens |
| KV allocation | 4.125 GiB per GPU |
| Maximum batched tokens | 2,048 |
| Maximum active sequences | 8 |
| Decode capture sizes | `4 8 12 16 20 24 28 32` |

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
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e DS41_SKIP_CHECKPOINT_SHA256=1 \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e DSV41_SOURCE=/workspace/Model/DeepSeek-V4.1-Flash-1017-Engram \
  -e DSV41_PACKED_DIR=/workspace/ds41-exl3-1017-engram \
  -- \
  vllm serve /workspace/Model/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --kv-cache-memory-bytes 4429185024 \
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
    --max-model-len 1048576 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 8 \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --max-cudagraph-capture-size 32 \
    --cudagraph-capture-sizes 4 8 12 16 20 24 28 32 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3}'
```

> [!WARNING]
> **1M Context and Memory Headroom**
>
> `--max-model-len 1048576` sets the per-sequence limit, including prompt and generation. It does not guarantee that every 1M-token workload fits, especially with images or concurrent requests.
>
> `--max-num-seqs 8` does not mean eight simultaneous full 1M-token requests can fit. Check memory on **both Sparks**, including during real requests.

---

## Engram SSD Settings

| Environment variable | Purpose |
| --- | --- |
| `DSV41_SOURCE` | Directory containing the original 1017 Engram configuration, index and two source shards. |
| `DSV41_PACKED_DIR` | Persistent, writable directory for each rank's packed Engram files. |
| `DS41_SKIP_CHECKPOINT_SHA256=1` | Skip full weight and existing packed-payload SHA256 scans during startup. |

The two directory variables originate from the community Engram SSD integration. The SHA256 skip switch is an image-specific addition.

> [!NOTE]
> SHA256 skipping retains metadata and file-size checks, but those checks cannot detect same-size payload corruption. Remove this variable or set it to `0` for full hash verification.
>
> This setting does not skip model loading, SSD reads or first-time packing, and does not change model precision.

---

## CUDA Graph Capture Sizes

For this recipe's regular DSpark verification step, each request contributes **K + 1 token rows**: K draft tokens and one target token.

```text
K = num_speculative_tokens = 3
S = max_num_seqs = 8

Capture sizes = (K + 1) × [1, 2, ..., S]
              = 4 8 12 16 20 24 28 32

Maximum capture size = (K + 1) × S = 32
```

**32 is the number of token rows in a captured execution, not the context length.** Context is controlled separately by `--max-model-len`.

If you change the speculative token count or maximum active requests, review the list and capture maximum together. These sizes cover regular verification batches in this recipe; other execution paths can have different shapes. Additional or larger captures may retain more graph memory.

---

## Memory Tuning Order

**Choose the execution settings first, then fit the KV budget, and lower the context limit only if the desired context cannot fit.**

The starting configuration is:

```bash
    --kv-cache-memory-bytes 4429185024 \
    --max-model-len 1048576 \
```

`4429185024` bytes is exactly **4.125 GiB per GPU**. Model weights, Engram lookup buffers, vision processing, CUDA graphs, communication and temporary workspaces consume additional memory from each Spark's shared physical memory.

1. **Choose concurrency, DSpark length, prefill budget and graph sizes.** Changes can affect memory headroom.
2. **Check startup KV capacity for the desired context.** If the capacity check says more cache is required and both nodes have physical headroom, increase `--kv-cache-memory-bytes` using the reported requirement as a guide.
3. **Distinguish a capacity error from a physical OOM.** For an actual OOM, do not increase the KV budget. To preserve 1M, first consider fewer/smaller captures, lower concurrency or a smaller prefill batch. Keep the capture list consistent with its maximum.
4. **If the desired context still cannot fit, reduce the explicit KV budget and context limit together as needed.** Use the capacity reported at startup to choose the context limit.

> [!IMPORTANT]
> Lowering only `--max-model-len` does **not** automatically reduce an explicitly fixed `--kv-cache-memory-bytes` allocation.
>
> With an explicit KV budget, increasing `--gpu-memory-utilization` does not enlarge that allocation. The utilization setting is not a hard cap on process or system memory use.
>
> Increasing the capture maximum does not automatically change the KV budget or context limit.

```text
More / larger graph captures
  → potentially more graph memory
  → less physical headroom for KV and request buffers

Insufficient KV capacity + physical headroom available
  → increase the explicit KV budget

Physical OOM
  → reduce graph/runtime memory
  → if necessary, reduce KV budget and context limit together
```

Source: [KV allocation in the pinned vLLM worker](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_worker.py).

---

## Notes

- This recipe is designed for **2× DGX Spark**. Both nodes must use the same image, EXL3 checkpoint revision and original Engram revision.
- **TP2 is enabled; DCP2 is not enabled** in this command.
- DSpark uses the weights included in the target checkpoint; no external draft model is required.
- EXL3 MoE/Linear computation uses ExLlamaV3. B12X NVFP4/MXFP8 flags are not interchangeable with this weight format.
- The EXL3 checkpoint is already quantized. It is not equivalent to the original full-precision model. SSD packing preserves the original Engram bytes without further quantization.
- The KV cache uses **FP8**, independently of EXL3 weight quantization.
- Vision remains enabled. Do not add `--language-model-only` if you need image inputs.
- No custom image-count restriction is added. Multimodal requests still need sufficient encoder and runtime memory.
- `--skip-mm-profiling` skips multimodal startup profiling; it does not remove image-processing memory costs.
- `mode: 0` disables torch.compile; `FULL_DECODE_ONLY` requests full decode CUDA Graphs.
- The explicit Engram row-cache budget defaults to **0 GiB**. Lookup buffers and other runtime allocations still consume memory.
- Adjust network interface, HCA, worker IP and local paths before starting. Keep the original Engram files and each node's packed directory on persistent local storage.

---

## Verify Server

```bash
curl "http://<HEAD_IP>:8000/v1/models"
```

OpenAI-compatible API base URL:

```text
http://<HEAD_IP>:8000/v1
```

Served model name:

```text
deepseek-v4.1-flash
```

---

## Test

### 1. llama-benchy

```
| model               |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:--------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| deepseek-v4.1-flash |         pp2048 | 1330.49 ± 306.71 |              | 1350.14 ± 365.50 | 1343.42 ± 365.50 | 1350.99 ± 365.51 |
| deepseek-v4.1-flash |          tg128 |     26.50 ± 3.58 | 37.33 ± 1.70 |                  |                  |                  |
| deepseek-v4.1-flash |         pp2048 | 1410.24 ± 260.57 |              | 1331.50 ± 285.54 | 1324.78 ± 285.54 | 1332.78 ± 285.58 |
| deepseek-v4.1-flash |         tg1024 |     30.54 ± 5.78 | 44.33 ± 8.34 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d1024 |  1326.56 ± 29.37 |              |  1998.53 ± 66.27 |  1991.82 ± 66.27 |  1999.80 ± 66.23 |
| deepseek-v4.1-flash |  tg128 @ d1024 |     32.12 ± 1.85 | 43.67 ± 5.56 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d1024 |  1367.69 ± 15.92 |              |  1916.76 ± 13.81 |  1910.04 ± 13.81 |  1917.96 ± 13.60 |
| deepseek-v4.1-flash | tg1024 @ d1024 |     31.43 ± 5.14 | 50.67 ± 4.78 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d4096 | 1486.39 ± 132.08 |              | 3501.11 ± 313.82 | 3494.40 ± 313.82 | 3501.11 ± 313.82 |
| deepseek-v4.1-flash |  tg128 @ d4096 |     28.24 ± 2.88 | 37.00 ± 1.63 |                  |                  |                  |
| deepseek-v4.1-flash | pp2048 @ d4096 |   1599.79 ± 7.86 |              |  3352.00 ± 31.39 |  3345.29 ± 31.39 |  3352.00 ± 31.39 |
| deepseek-v4.1-flash | tg1024 @ d4096 |     29.40 ± 2.40 | 49.67 ± 2.49 |                  |                  |                  |

llama-benchy (0.4.1.dev1+ge9be34457)
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

---

## Sources

- [vLLM 0.32.0](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer/tree/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1)
- [ExLlamaV3 source](https://github.com/turboderp-org/exllamav3/tree/e648f1a131365aae15920073e761a3fa5a527654)
- [EXL3 / two-Spark integration kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452)
- [eugr Spark Docker reference](https://github.com/eugr/spark-vllm-docker/tree/0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d)
- [EXL3 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/0be63c3337efbb2a0061d5b8bb4006fa186d433f)
- [Original 1017 checkpoint / Engram](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
