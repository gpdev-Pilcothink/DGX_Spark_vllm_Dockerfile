# Solar-Open2-250B-Nota-INT4 on 2× DGX Spark

This recipe runs **Solar-Open2-250B-Nota-INT4** on **2× DGX Spark** using vLLM.

Maintained by **Pilcothink**.

This recipe uses a prebuilt Docker image. No Docker build is required.

---

## Prerequisites

> [!IMPORTANT]
> If you are using **two or more DGX Spark systems** and have not yet configured your cluster, first complete the [Multi-Node Cluster Setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/README.md#multi-node-cluster-setup) section in the main README.
>
> This recipe assumes that the cluster setup and communication tests described there have been completed successfully.

---

## Models

### Target Model

**nota-ai/Solar-Open2-250B-Nota-INT4**

https://huggingface.co/nota-ai/Solar-Open2-250B-Nota-INT4

---

## Step 1. Download the Model

Download the target model on **both DGX Spark systems**.

```bash
hf download nota-ai/Solar-Open2-250B-Nota-INT4 \
  --local-dir /path/Model/Solar-Open2-250B-Nota-INT4
```

Example directory structure:

```text
/path/
├── Model/
│   └── Solar-Open2-250B-Nota-INT4/
└── hf_cache/
```

---

## Step 2. Pull Docker Image

Run on **both DGX Spark systems**.

```bash
docker pull pilcothink/upstage_spark:0.22
```

## run_cluster_dual.sh

Download `run_cluster_dual.sh` from:

https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/main/run_cluster_dual.sh

---

## Step 3. Configure Dual DGX Spark

Run on the **head DGX Spark**.

```bash
cd /path/to/vllm-recipe-repository

export VLLM_IMAGE=pilcothink/upstage_spark:0.22

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

## Step 4. Run Solar-Open2

Run on the **head DGX Spark only**.

```bash
bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HF_CACHE_DIR" \
  --no-ray \
  --workers "$WORKER_NODE_IP" \
  --master-port 29501 \
  --ipc=host --privileged \
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
  -- \
  vllm serve /workspace/Model/Solar-Open2-250B-Nota-INT4 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 \
    --served-model-name solar-open2-250b \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --reasoning-parser solar_open2 \
    --tool-call-parser solar_open2 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --moe-backend triton \
    --max-num-batched-tokens 8192 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 10 \
    --default-chat-template-kwargs '{"think_render_option":"preserved"}' \
    --logits-processors vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor
```

---

## Notes

* This recipe is designed for **2× DGX Spark**.
* Both nodes must use the same Docker image.
* Both nodes must have the same target model at the same configured path.
* The target KV cache uses FP8.
* The MoE backend uses Triton.
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