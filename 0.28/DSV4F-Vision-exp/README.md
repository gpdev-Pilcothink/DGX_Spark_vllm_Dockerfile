<Build Command>
docker build -t vllm_spark_dsv4:0.29-b12x




<serve command>

Refer to the environment variable settings in the following recipe:

https://github.com/eugr/spark-vllm-docker/blob/main/recipes/deepseek-v4-flash-0731.yaml

However, change `VLLM_USE_AOT_COMPILE` and `VLLM_USE_BREAKABLE_CUDAGRAPH` as follows:

```bash
export VLLM_USE_AOT_COMPILE=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

Then, use the following serving command:


vllm serve /DeepSeek-V4-Flash-Vision-Exp \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
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
    --speculative-config '{"method":"dspark","model":"/DeepSeek-V4-Flash-Vision-Exp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","attention_backend":"FLASHINFER_MLA_SPARSE_DSV4","enable_adaptive_verification":false}'