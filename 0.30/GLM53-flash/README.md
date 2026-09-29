# GLM-5.3-Flash NVFP4 on 2× DGX Spark

This recipe runs **NVIDIA GLM-5.3-Flash-NVFP4** on **2× DGX Spark** using **vLLM 0.30**.

Two speculative decoding modes are available:

* **MTP**
* **DFlash2** — the default recipe, with **1,048,576 tokens (1M context)** and **4 speculative tokens**.

---

## Prerequisites

> [!IMPORTANT]
> If you are using **two or more DGX Spark systems** and have not yet configured your cluster, first complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup) section in the main README.
>
> This recipe assumes that the cluster setup and communication tests described there have been completed successfully.

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

<!-- Release preparation: publish pilcothink/vllm_spark_glm53:0.30 before publishing this README. -->

Run on **both DGX Spark systems**.

```bash
docker pull pilcothink/vllm_spark_glm53:0.30
```


## run_cluster_dual.sh

Download `run_cluster_dual.sh` from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh


---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_glm53:0.30

export MN_IF_NAME=enp1s0f1np1
export NCCL_HCA=rocep1s0f1
export WORKER_NODE_IP="<WORKER_IP>"

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

This alternative retains a 262,144-token context setting. For the default 1M-context configuration, use **Option B. DFlash2** below.

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

The default configuration uses **1,048,576 tokens**, **4 speculative tokens**, and **up to 8 concurrent sequences**.

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
  -e GLM53_GROUPED_CONTEXT_STORE=1 \
  -e GLM53_FUSED_DFLASH_TAPS=1 \
  -- \
  vllm serve /workspace/Model/GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --moe-backend b12x \
    --linear-backend b12x \
    --gpu-memory-utilization 0.88 \
    --kv-cache-memory-bytes 9600000000 \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-model-len 1048576 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-cudagraph-capture-size 40 \
    --cudagraph-capture-sizes 5 10 15 20 25 30 35 40 \
    --max-num-seqs 8 \
    --speculative-config '{"method":"dflash","model":"/workspace/Model/GLM-5.3-Flash-DFlash2","num_speculative_tokens":4,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"disable_eagle_block_drop":false}'
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
"num_speculative_tokens":4
```

The explicit CUDA capture list above is for **K=4**. When testing another draft length, update `num_speculative_tokens` and adjust the capture sizes and capture maximum together; the K=4 list should not be reused unchanged.

## CUDA Graph Capture Sizes

For this recipe's fixed-length DFlash decode, each active request contributes **K + 1 tokens** to a target verification step: K draft tokens plus one target token. To capture an exact size for every active request count from 1 through `--max-num-seqs`, use:

```text
K = num_speculative_tokens
S = max_num_seqs

Capture sizes = (K + 1) × [1, 2, ..., S]
Maximum capture size = (K + 1) × S

K = 4, S = 8
Capture sizes = 5 10 15 20 25 30 35 40
Maximum capture size = 40
```

| DFlash K | Maximum active requests | Capture sizes | Maximum capture size |
| --- | --- | --- | --- |
| 4 | 8 | `5 10 15 20 25 30 35 40` | 40 |
| 5 | 8 | `6 12 18 24 30 36 42 48` | 48 |
| 6 | 8 | `7 14 21 28 35 42 49 56` | 56 |
| 7 | 8 | `8 16 24 32 40 48 56 64` | 64 |

For the default K=4, S=8 recipe:

```bash
    --max-num-seqs 8 \
    --max-cudagraph-capture-size 40 \
    --cudagraph-capture-sizes 5 10 15 20 25 30 35 40 \
```

**40 is the number of token rows in one captured execution, not the context length.** The 1M context limit is controlled separately by `--max-model-len 1048576`. Keep the capture maximum equal to the largest entry in the explicit list; changing only the maximum leaves the two settings inconsistent.

This list covers all eight regular decode batch sizes in the current recipe, with adaptive verification and sequence parallelism disabled. It is not a universal requirement for other execution modes. Relative to this build's default decode sizes `5 10 20 25 35 40`, adding 15 and 30 avoids padding three requests from 15 to 20 rows and six requests from 30 to 35 rows. Additional captures can retain more graph memory; this does not guarantee a proportional throughput gain.

Source: [vLLM decode graph dispatcher at the pinned commit](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/cudagraph_dispatcher.py).

## Memory Tuning Order

**Choose the graph configuration first, then fit the KV cache budget, and lower the context limit only if 1M cannot fit.** The default DFlash2 command starts with:

```bash
    --kv-cache-memory-bytes 9600000000 \
    --max-model-len 1048576 \
```

The memory budget on each GPU must cover model weights, runtime buffers and workspaces, CUDA graphs, KV cache, and remaining headroom. Larger or additional captures can increase graph memory and leave less room for KV cache. The cost depends on the captured execution and memory sharing; there is no fixed bytes-per-capture-size conversion.

**Increasing `--max-cudagraph-capture-size` does not automatically increase `--kv-cache-memory-bytes` or decrease `--max-model-len`.** Both explicit values stay unchanged. Under a fixed memory budget, larger graphs may require a smaller KV allocation, which can in turn force a shorter supported context.

1. **Choose DFlash K, maximum active requests, and capture sizes.** Update the list and maximum together using the calculation above. Recheck memory after changing them.
2. **Keep the desired context at 1,048,576 tokens and check startup KV capacity.** Start from the recipe's 9,600,000,000-byte KV budget per GPU. This is a starting setting, not a guarantee of capacity under every graph configuration.
3. **If the KV capacity check reports that 1M needs more cache, and both nodes have memory headroom, increase `--kv-cache-memory-bytes`.** Use the reported requirement as a guide, retain runtime headroom, and restart to check capacity again. With an explicit KV budget, increasing `--gpu-memory-utilization` does not increase that allocation.
4. **If CUDA reports an actual out-of-memory error, do not increase the KV budget.** To preserve 1M, first reduce graph memory by using fewer capture sizes or a lower capture maximum, and consider lowering concurrency. Keep the explicit list consistent with its maximum; a smaller set may introduce padding or leave some batches outside graph coverage.
5. **If 1M still cannot fit, reduce the KV budget as needed and lower `--max-model-len` to the capacity reported at startup.** Lowering only `--max-model-len` does not automatically shrink an explicitly fixed KV allocation. Increasing `--max-model-len` is not an OOM fix.

```text
Larger / more captures
  → potentially more graph memory
  → less memory available for KV cache
  → potentially a lower affordable context limit

KV capacity too small + physical headroom available
  → increase the explicit KV budget to preserve 1M

Physical OOM
  → reduce graph/runtime memory first to preserve 1M
  → if insufficient, reduce KV budget and context length together
```

Check both Spark nodes and allow headroom for serving, not just startup. Eight active requests does not mean eight full 1M contexts fit simultaneously.

Sources: [explicit KV allocation in the pinned vLLM worker](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_worker.py), [KV capacity validation](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/core/kv_cache_utils.py), and [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/).

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
* The default DFlash2 recipe uses `--max-model-len 1048576`; no separate 1M setup is required.
* `--kv-cache-memory-bytes 9600000000` reserves approximately **8.94 GiB per GPU** for KV cache.
* `--max-num-seqs 8` does not imply capacity for eight simultaneous full 1M-token requests.
* The DFlash2 command explicitly enables **Grouped Context Store** and **Fused DFlash Taps**. Both image defaults are `0`; change either command value to `0` to disable it.
* Fused Taps checks its intermediate and final results against the native kernel before first use. If the check fails, restart with `GLM53_FUSED_DFLASH_TAPS=0`.
* The capture list `5 10 15 20 25 30 35 40` includes exact decode sizes for one to eight requests with K=4. Extra captures may retain additional graph memory.
* DFlash2 uses `TRITON_ATTN` for the draft attention backend.
* DFlash2 supports `num_speculative_tokens` values from **4 to 7** in this Docker image.
* `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, worker IP, and local paths must be adjusted for your environment.

---

## Verify Server

```bash
curl "http://<HEAD_IP>:8000/v1/models"
```

The OpenAI-compatible API is available at:

```text
http://<HEAD_IP>:8000
```


---

## ===========Test===========

```
| model                                |           test |              t/s |     peak t/s |       ttfr (ms) |    est_ppt (ms) |   e2e_ttft (ms) |
|:-------------------------------------|---------------:|-----------------:|-------------:|----------------:|----------------:|----------------:|
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1409.02 ± 20.36 |              | 1206.90 ± 12.93 | 1199.84 ± 12.93 | 1206.90 ± 12.93 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |          tg128 |     35.23 ± 4.80 | 41.33 ± 4.99 |                 |                 |                 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1440.21 ± 46.66 |              | 1190.72 ± 25.69 | 1183.65 ± 25.69 | 1190.72 ± 25.69 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         tg1024 |     31.17 ± 1.46 | 49.00 ± 2.94 |                 |                 |                 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  1351.92 ± 15.39 |              | 1963.52 ± 25.94 | 1956.46 ± 25.94 | 1963.52 ± 25.94 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d1024 |     33.06 ± 1.37 | 41.67 ± 0.94 |                 |                 |                 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 |  1340.60 ± 12.04 |              | 1928.65 ± 62.03 | 1921.59 ± 62.03 | 1928.65 ± 62.03 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d1024 |     34.35 ± 4.29 | 51.00 ± 3.56 |                 |                 |                 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 | 1744.31 ± 178.75 |              | 3149.61 ± 77.47 | 3142.55 ± 77.47 | 3149.61 ± 77.47 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d4096 |     32.30 ± 1.67 | 44.67 ± 5.56 |                 |                 |                 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 |   1603.89 ± 4.84 |              | 3233.26 ± 40.45 | 3226.19 ± 40.45 | 3233.26 ± 40.45 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d4096 |     34.46 ± 3.29 | 52.67 ± 1.25 |                 |                 |                 |

llama-benchy (0.4.1.dev1+ge9be34457)
```