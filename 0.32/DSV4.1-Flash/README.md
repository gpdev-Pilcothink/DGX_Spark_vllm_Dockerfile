## Build Command

DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw with original 1017 Engram on local NVMe, for **2× DGX Spark**.

Built from official vLLM **0.32.0**, CUDA **13.0.2**, FlashInfer **0.7.0.post1**, and ExLlamaV3, with the EXL3/Engram integration and local compatibility fixes included in the Dockerfile.

Run in the directory containing this Dockerfile:

```bash
docker build -t vllm_ds41:0.32-exl3 .
```

The Dockerfile is self-contained. Existing Docker layer cache is reused when its inputs match. Model weights are downloaded separately and are not included in the image.

### Docker Image — Push / Pull

Publish the locally built image:

```bash
docker login
docker tag vllm_ds41:0.32-exl3 pilcothink/vllm_ds41:0.32-exl3
docker push pilcothink/vllm_ds41:0.32-exl3
```

On **both Sparks**, pull the published image and assign the local tag used by the serving command:

```bash
docker pull pilcothink/vllm_ds41:0.32-exl3
docker tag pilcothink/vllm_ds41:0.32-exl3 vllm_ds41:0.32-exl3
```

Both nodes must use the same image digest. The Docker Hub tag becomes available after the push completes.

## Serve Command

### Model Files

Prepare the following files on **each node's local NVMe**. The Dockerfile checks the pinned checkpoint revisions below.

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

### Two-Node Launch

Use your existing `run_cluster_dual.sh` in `~/Desktop/storage/DGX_Spark/vllm`. Adjust the interface, HCA, worker IP and host paths for your environment. Both nodes must expose the same container model paths.

```bash
cd ~/Desktop/storage/DGX_Spark/vllm

VLLM_IMAGE=vllm_ds41:0.32-exl3
MN_IF_NAME=enp1s0f1np1
VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')
WORKER_NODE_IP=169.254.139.122

echo "Head IP: $VLLM_HOST_IP"
echo "Worker IP: $WORKER_NODE_IP"

bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HOME/Desktop/AI/vllm/hf_cache" \
  --no-ray \
  --workers "$WORKER_NODE_IP" \
  --master-port 29501 \
  --ipc=host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/pil/Desktop/AI:/workspace \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e NCCL_IB_HCA=rocep1s0f1 \
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

### Runtime Notes

- **TP2, DCP1:** the command does not enable decode context parallelism. DCP2 is a separate optional custom path and is not part of this serving recipe.
- **EXL3 computation:** MoE/Linear use the EXL3 integration with ExLlamaV3. Do not replace these flags with B12X NVFP4/MXFP8 backends; they use different weight formats.
- **Original Engram on SSD:** `DSV41_SOURCE` points to the original shards; `DSV41_PACKED_DIR` is the writable, persistent directory for rank-specific packed tables. The image enables packing and defaults the explicit row cache to `0 GiB`. This does not eliminate working buffers or OS memory use.
- **First startup:** each node packs its own Engram rows without changing their stored precision. Allow approximately **95 GiB of additional SSD space per node** for packed files. Keep the original shards and the packed directory between restarts.
- **SHA256:** `DS41_SKIP_CHECKPOINT_SHA256=1` skips full weight/packed-payload SHA256 scans. Metadata and file sizes remain checked, but same-size corruption is not detected by those checks. Remove this variable or set it to `0` for full hash verification. Packing and model loading still read the required data.
- **KV budget:** `4429185024` bytes is exactly **4.125 GiB per GPU**. This explicit setting controls KV cache allocation instead of deriving its size from `--gpu-memory-utilization`; that utilization value is not a hard process-memory limit.
- **Context:** the recipe requests **1,048,576 tokens per sequence**, including prompt and generation. Successful full-length requests also require sufficient runtime memory. `--max-num-seqs 8` does not promise eight simultaneous 1M-token requests.
- **Decode captures:** DSpark 3 corresponds to a target verification width of `3 + 1 = 4` tokens per request. The explicit sizes `4 8 12 16 20 24 28 32` cover those widths for one through eight requests. Additional captures retain graph memory; these values are serving arguments, not Dockerfile resource defaults.
- **Vision:** vision support remains enabled. No custom image-count limit is added by this command. `--skip-mm-profiling` skips multimodal startup profiling; it does not remove the memory needed by actual image requests.
- **Compilation:** `mode: 0` disables torch.compile while `FULL_DECODE_ONLY` requests full decode CUDA Graphs.

## Test

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

- [vLLM 0.32.0](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer/tree/946200de1ae94fc93fdd0926f0a13afd1fa7f0f1)
- [ExLlamaV3 source](https://github.com/turboderp-org/exllamav3/tree/e648f1a131365aae15920073e761a3fa5a527654)
- [EXL3 / two-Spark integration kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452)
- [eugr Spark Docker reference](https://github.com/eugr/spark-vllm-docker/tree/0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d)
- [EXL3 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/0be63c3337efbb2a0061d5b8bb4006fa186d433f)
- [Original 1017 checkpoint / Engram](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)


This is a custom integration built on official sources, not an unmodified upstream vLLM image. Preserve the bundled source, patches and license notices when redistributing it; the integration kit is AGPL-3.0. Build provenance is stored under `/workspace/provenance` in the image.
