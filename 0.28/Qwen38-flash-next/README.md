# Qwen3.8-Flash-Next-NVFP4 — Single DGX Spark Run Recipe

This recipe is for running **NVIDIA Qwen3.8-Flash-Next-NVFP4** on a **single DGX Spark** using vLLM.

Model:

https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4

Docker Image:

```text
pilcothink/vllm_spark_qwen38:0.28
```

---

## Step 1. Download the Model

Download the model from Hugging Face:

https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4

Place the downloaded model inside the local directory that will be mounted to `/workspace`.

For example:

```text
/path/Qwen3.8-Flash-Next-NVFP4
```

will be accessible inside the container as:

```text
/workspace/Qwen3.8-Flash-Next-NVFP4
```

---

## Step 2. Pull the Docker Image

```bash
docker pull pilcothink/vllm_spark_qwen38:0.28
```

---

## Step 3. Run the Container

```bash
docker run -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v /path:/workspace \
  -v /path/hf_cache:/root/.cache/huggingface \
  -e VLLM_PLE_SSD=1 \
  pilcothink/vllm_spark_qwen38:0.28
```

> **Note**
>
> Replace `/path` with your own local directory.

Example:

```bash
-v /home/user/AI:/workspace \
-v /home/user/AI/hf_cache:/root/.cache/huggingface \
```

---

## Step 4. Start the vLLM Server

Inside the container, run:

```bash
vllm serve /workspace/Qwen3.8-Flash-Next-NVFP4 \
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

The OpenAI-compatible vLLM API server will be available on:

```text
http://localhost:8000
```

---

## Configuration Summary

| Option | Value |
|---|---|
| Target | Single DGX Spark |
| Model | Qwen3.8-Flash-Next-NVFP4 |
| Tensor Parallel | 1 |
| Linear Backend | B12X |
| MoE Backend | B12X |
| CUDA Graph | FULL_DECODE_ONLY |
| Max Context Length | 262,144 |
| Max Batched Tokens | 4,096 |
| Max Sequences | 2 |
| Prefix Caching | Enabled |
| Chunked Prefill | Enabled |
| Speculative Decoding | MTP |
| MTP Speculative Tokens | 3 |
| Reasoning Parser | qwen3 |
| Tool Call Parser | qwen3_xml |
| API Port | 8000 |