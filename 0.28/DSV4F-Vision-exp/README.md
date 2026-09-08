## Build Command

```bash
docker build -t vllm_spark_dsv4:0.29-b12x .
```

## Serve Command

Refer to the environment variable settings in the following recipe:

https://github.com/eugr/spark-vllm-docker/blob/main/recipes/deepseek-v4-flash-0731.yaml

However, set the following environment variables as shown below:

```bash
export VLLM_USE_AOT_COMPILE=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

Then, start the server using the following command:

```bash
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
```

## Test

**1. llama-benchy (tg 128, 1024)**

```
| model                                         |           test |              t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:----------------------------------------------|---------------:|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 |  1901.57 ± 64.87 |              |   938.12 ± 28.27 |   934.13 ± 28.27 |   938.12 ± 28.27 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |          tg128 |     40.78 ± 2.35 | 46.67 ± 2.49 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 |   1851.57 ± 1.50 |              |   969.66 ± 11.07 |   965.66 ± 11.07 |   969.66 ± 11.07 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         tg1024 |     39.27 ± 3.25 | 56.67 ± 7.59 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 |  1928.68 ± 20.71 |              |  1389.57 ± 20.04 |  1385.57 ± 20.04 |  1389.57 ± 20.04 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d1024 |     39.16 ± 0.78 | 45.00 ± 1.63 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 |  1912.42 ± 65.02 |              |  1395.47 ± 51.24 |  1391.48 ± 51.24 |  1395.47 ± 51.24 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d1024 |     39.41 ± 3.77 | 59.00 ± 2.94 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 |  2064.76 ± 14.24 |              |   2531.87 ± 9.12 |   2527.88 ± 9.12 |   2531.87 ± 9.12 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d4096 |     40.50 ± 3.20 | 46.00 ± 2.16 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 | 1921.19 ± 197.40 |              | 2781.11 ± 368.86 | 2777.12 ± 368.86 | 2781.11 ± 368.86 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d4096 |     39.69 ± 1.09 | 57.67 ± 1.89 |                  |                  |                  |

llama-benchy (0.3.9.dev9+g446dd42fd)
date: 2026-09-08 13:09:10 | latency mode: api
```

**2. tool-eval-bench v2.6.1.dev57+g9ab686613 --hardmode**

```
tool-eval-bench --backend vllm --base-url http://127.0.0.1:8000 --seed 42 --hardmode

╭─────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────╮
│                                                                                                                      │
│    Model:  /workspace/Model/DeepSeek-V4-Flash-Vision-Exp                                                             │
│    Score:  93 / 100                                                                                                  │
│    Rating: ★★★★★ Excellent                                                                                           │
│    Benchmark: tool-eval-bench v2.6.1.dev57+g9ab686613                                                                │
│    Engine:       vLLM 0.28.1rc1.dev475+g6fbb00b18.d20260907                                                          │
│    Max context:  1,048,576 tokens                                                                                    │
│                                                                                                                      │
│    ✅ 75 passed   ⚠️  13 partial   ❌ 0 failed                                                                       │
│    Points: 163/176                                                                                                   │
│                                                                                                                      │
│    Quality:        93/100                                                                                            │
│    Responsiveness: 46/100  (median turn: 3.3s)                                                                       │
│    Deployability:  79/100  (α=0.7)                                                                                   │
│    Weakest: E Error Recovery (83%)                                                                                   │
│                                                                                                                      │
│    Completed in 1286.1s                                                                                              │
│                                                                                                                      │
│    📊 Token Usage:                                                                                                   │
│    Total: 630,094 tokens  │  Efficiency: 0.3 pts/1K tokens                                                           │
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



**3. MCP PDF Analysis Test** 
A seven-page PDF containing science-based knowledge presented through a variety of illustrations, diagrams, charts, and images was provided for analysis. The PDF consists entirely of image-based content, with no OCR or selectable text applied.

The task was to analyze the contents of the PDF, conduct additional research on the Internet based on the information presented, and then create a Word document that organizes and expands upon the findings.


<img width="438" height="497" alt="스크린샷 2026-09-08 135439" src="https://github.com/user-attachments/assets/5afeb816-db71-46e7-a806-4f3941240dec" /> <img width="438" height="497" alt="스크린샷 2026-09-08 140007" src="https://github.com/user-attachments/assets/442046e0-8d9b-4e90-adbf-795f8f7639d7" />
<img width="709" height="241" alt="스크린샷 2026-09-08 140013" src="https://github.com/user-attachments/assets/b299c741-38e9-4c3c-aea1-4e054044dc32" />








**4. MCP OFFICE Job Test**
The following is the Word document output from MCP Task 3.
<img width="576" height="421" alt="MCP OFFICE Job Test" src="https://github.com/user-attachments/assets/3aa126e5-077d-44b1-ba08-63218371469a" />


