# ERNIE-5.5-Flash-Preview on two DGX Sparks

This image runs ERNIE's sparse attention and routed experts with B12X on two GB10 GPUs (TP2), using vLLM 0.34.0 and Baidu's model plugin. The checkpoint supplies blockwise FP8 projections, MXFP4 routed experts and its DSpark draft head.

> [!IMPORTANT]
> **Complete the [multi-node cluster setup](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile#multi-node-cluster-setup) and verify NCCL communication between the two Sparks before serving.**

> [!WARNING]
> **Download [run_cluster_dual.sh](https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/blob/b13f064cb3e6e04c2969e981f07717b220005db3/run_cluster_dual.sh) separately on the head Spark.** Pulling the image does not install this host-side script. Run the command below from its directory.

## Model

Download the same revision on both nodes. The FP4-expert checkpoint occupies approximately 155 GiB of storage per node. Keep its tokenizer, Python files and `flashindexer/` directory alongside the shards.

```bash
MODEL_ROOT="$HOME/AI/Model"
hf download baidu/ERNIE-5.5-Flash-Preview \
  --revision ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe \
  --local-dir "$MODEL_ROOT/ERNIE-5.5-Flash-Preview"
```

## Image

Once published, pull the same release image on both Sparks.

```bash
docker pull pilcothink/vllm_spark:0.34-ernie55
```

Optionally run the component check on each Spark before loading weights. It checks DSA and compressed attention against known outputs, compares router IDs and FP32 weights bit for bit with Baidu's router, prints router timings, and compares the native W4A8 SiTU output with the MXFP8 numerical reference.

```bash
docker run --rm --gpus all pilcothink/vllm_spark:0.34-ernie55 python3 /opt/ernie/check_b12x.py
```

## Download the launcher on the head

```bash
mkdir -p "$HOME/dgx-spark-vllm"
cd "$HOME/dgx-spark-vllm"
curl -fL https://raw.githubusercontent.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/b13f064cb3e6e04c2969e981f07717b220005db3/run_cluster_dual.sh -o run_cluster_dual.sh
```

## Start on the head

Set the fabric interface, HCA and worker address for your cluster. Both nodes need the same host mount paths. This profile allows up to 262,144 tokens per request and drafts four tokens per DSpark step. Up to 10 sequences can be active. Their combined capacity depends on the KV cache reported at startup.

```bash
cd "$HOME/dgx-spark-vllm"
VLLM_IMAGE=pilcothink/vllm_spark:0.34-ernie55
MN_IF_NAME='<FABRIC_INTERFACE>'
NCCL_HCA='<RDMA_HCA>'
WORKER_NODE_IP='<WORKER_FABRIC_IP>'
MODEL_ROOT="$HOME/AI/Model"
HF_CACHE_DIR="$HOME/AI/vllm/hf_cache"
VLLM_HOST_IP=$(ip -4 -o addr show "$MN_IF_NAME" | awk 'NR==1 {split($4,a,"/"); print a[1]}')

bash run_cluster_dual.sh "$VLLM_IMAGE" "$VLLM_HOST_IP" \
  --head "$HF_CACHE_DIR" --no-ray --workers "$WORKER_NODE_IP" \
  --master-port 29501 --ipc=host --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_ROOT:/models:ro" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" -e NCCL_IB_HCA="$NCCL_HCA" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -- \
  vllm serve /models/ERNIE-5.5-Flash-Preview \
    --served-model-name baidu/ERNIE-5.5-Flash-Preview \
    --host 0.0.0.0 --port 8000 \
    --distributed-executor-backend mp --tensor-parallel-size 2 \
    --trust-remote-code --dtype bfloat16 --load-format safetensors \
    --moe-backend b12x --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 262144 --max-num-seqs 10 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.875 \
    --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice \
    --speculative-config '{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
```

The expected expert backend in the startup log is `B12X_MXFP4_MXFP8`. It uses Baidu's SiTU activation (beta 4, linear beta 25) and TP-sharded experts. EP is disabled. FP8 dense projections use the engine's linear backend selection. The DSA indexer uses the plugin's FP8 path on Spark.

The ERNIE plugin selects B12X directly for the 37 HCA layers and six DSA layers. HCA uses B12X single-pass attention for prefill and split attention for decode, with one normalization across the window, compressed history and learned sink. DSA uses the B12X split kernel for the complete 2,176-slot index row. The vLLM base classes supply the KV layout and scheduling metadata. Leave the global attention flag unset for this model.

## Sources

- [Baidu checkpoint and model card](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview/tree/ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe)
- [Baidu ERNIE model and plugin](https://github.com/baidu/vLLM-ERNIE/tree/ba322a5634f91d3e6f96ae711df72ddaf3c38b6e)
- [Official vLLM 0.34.0](https://github.com/vllm-project/vllm/tree/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca)
- [Official FlashInfer 0.7.3](https://github.com/flashinfer-ai/flashinfer/tree/775207deb34055667df89ed618fb4b06fc84de74)
- [eugr B12X preset](https://github.com/eugr/spark-vllm-docker/blob/b288d0947791c37c4db8e68b8709156f6a548814/build-and-copy.sh)
- [B12X provider at eugr's selected revision](https://github.com/local-inference-lab/flashinfer/tree/9873d7a33f91aeb14c83286552e1f3d84efa2c2e/flashinfer/experimental/b12x)
