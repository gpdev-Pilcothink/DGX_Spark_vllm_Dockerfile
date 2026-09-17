# GLM-5.3-Flash NVFP4 on 2× DGX Spark

This recipe runs **NVIDIA GLM-5.3-Flash-NVFP4** on **2× DGX Spark** using vLLM.

Two speculative decoding modes are available:

* **MTP**
* **DFlash2**

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

Run on **both DGX Spark systems**.

### vLLM 0.29

```bash
docker pull pilcothink/vllm_spark_glm53:0.29
```

### Previous vLLM 0.28 Image

The previous vLLM 0.28 image remains available:

```bash
docker pull pilcothink/vllm_spark_glm53:0.28
```

The `0.29` image is recommended for the latest GLM-5.3-Flash configuration and DGX Spark optimizations.

## run_cluster_dual.sh

Download `run_cluster_dual.sh` from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh

---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/vllm_spark_glm53:0.29

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
  -e GLM53_CONTEXT_CUDA_GRAPH=1 \
  -e GLM53_GROUPED_CONTEXT_STORE=1 \
  -e GLM53_CONTEXT_GRAPH_CACHE_V2=0 \
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
  -e GLM53_CONTEXT_CUDA_GRAPH=1 \
  -e GLM53_GROUPED_CONTEXT_STORE=1 \
  -e GLM53_FUSED_DFLASH_TAPS=1 \
  -e GLM53_CONTEXT_GRAPH_CACHE_V2=0 \
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

---

## vLLM 0.29 DGX Spark Optimizations

The `0.29` image adds additional GLM-5.3-Flash optimizations for DGX Spark.

The recommended environment configuration is:

```bash
-e GLM53_CONTEXT_CUDA_GRAPH=1 \
-e GLM53_GROUPED_CONTEXT_STORE=1 \
-e GLM53_FUSED_DFLASH_TAPS=1 \
-e GLM53_CONTEXT_GRAPH_CACHE_V2=0 \
```

These options are included in the example commands above.

The default recipe continues to use:

```bash
--max-model-len 262144
```

for additional memory headroom and concurrency.

The optional **1M context configuration** described below remains separate and should only be used when explicitly serving very long contexts.


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

1M context serving is supported on **2× DGX Spark**, but **DFlash2 is required for the 1M-token configuration in this recipe**.

The MTP mode uses the MTP / NextN weights included in the target checkpoint and does not leave enough memory headroom for the KV cache required to serve the full 1M-token context on two DGX Spark systems.

For 1M context serving, use the **DFlash2 configuration** and manually allocate a 9 GiB KV cache:

```bash
--gpu-memory-utilization 0.9 \
--kv-cache-memory=9663676416 \
--max-model-len 1000000 \
--speculative-config '{"method":"dflash","model":"/workspace/Model/GLM-5.3-Flash-DFlash2","num_speculative_tokens":5,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"disable_eagle_block_drop":false}'
```

`--kv-cache-memory=9663676416` allocates **9.0 GiB** of KV cache per worker.

> **Important:** For 1M context serving, both DGX Spark systems should have at least approximately **118 GiB of available memory** before starting the server.
>
> Check the available memory on both nodes with:
>
> ```bash
> free -h
> ```
>
> Make sure the `available` column reports at least **118 GiB** on each DGX Spark.

The default recipe uses a `262144` context length for a larger memory margin and higher concurrency. Use the DFlash2 configuration above when serving the full **1M-token context**.


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

## ===========1M Token Serving Test===========

```
| model                                |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:-------------------------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 | 1521.65 ± 149.99 |              |  1235.24 ± 38.03 |  1229.81 ± 38.03 |  1235.24 ± 38.03 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |          tg128 |     39.26 ± 6.95 | 51.67 ± 5.44 |                  |                  |                  |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         pp2048 |  1422.49 ± 33.97 |              |  1223.76 ± 14.71 |  1218.33 ± 14.71 |  1223.76 ± 14.71 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |         tg1024 |     29.43 ± 0.53 | 58.67 ± 1.25 |                  |                  |                  |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 | 1452.02 ± 127.95 |              |  1917.75 ± 27.85 |  1912.32 ± 27.85 |  1917.75 ± 27.85 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d1024 |     31.80 ± 1.32 | 43.33 ± 3.30 |                  |                  |                  |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d1024 | 1367.99 ± 260.96 |              | 2073.47 ± 312.29 | 2068.04 ± 312.29 | 2073.47 ± 312.29 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d1024 |     29.14 ± 1.91 | 51.67 ± 3.86 |                  |                  |                  |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 |   1397.12 ± 6.85 |              |  3646.64 ± 71.23 |  3641.21 ± 71.23 |  3646.64 ± 71.23 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 |  tg128 @ d4096 |     34.90 ± 1.90 | 45.67 ± 5.25 |                  |                  |                  |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | pp2048 @ d4096 |  1419.06 ± 13.54 |              |  3678.72 ± 72.57 |  3673.29 ± 72.57 |  3678.72 ± 72.57 |
| /workspace/Model/GLM-5.3-Flash-NVFP4 | tg1024 @ d4096 |     33.04 ± 4.23 | 53.00 ± 7.79 |                  |                  |                  |
```