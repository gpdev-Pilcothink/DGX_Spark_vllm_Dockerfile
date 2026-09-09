## Build Command

```bash
docker build --build-arg ENABLE_B12X=1 -t vllm_spark_dsv4:0.29-b12x .
```

## Serve Command

Refer to the environment variable settings in the following recipe:

https://github.com/eugr/spark-vllm-docker/blob/main/recipes/deepseek-v4-flash-0731.yaml

However, set the following environment variables as shown below:

```bash
export VLLM_USE_AOT_COMPILE=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_B12X_MOE_FP4_FORCE_A16=1
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
| model                                         |           test |             t/s |     peak t/s |        ttfr (ms) |     est_ppt (ms) |    e2e_ttft (ms) |
|:----------------------------------------------|---------------:|----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 | 1770.32 ± 25.26 |              |    999.94 ± 9.99 |    996.02 ± 9.99 |    999.94 ± 9.99 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |          tg128 |    44.06 ± 3.46 | 51.67 ± 6.24 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         pp2048 | 1773.69 ± 13.95 |              |    998.86 ± 9.20 |    994.94 ± 9.20 |    998.86 ± 9.20 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |         tg1024 |    42.70 ± 2.26 | 62.67 ± 4.50 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 |  1851.01 ± 6.32 |              |   1446.04 ± 7.18 |   1442.12 ± 7.18 |   1446.04 ± 7.18 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d1024 |    43.39 ± 1.73 | 54.33 ± 3.68 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d1024 | 1817.56 ± 33.58 |              |  1447.18 ± 39.04 |  1443.26 ± 39.04 |  1447.18 ± 39.04 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d1024 |    46.33 ± 5.96 | 62.00 ± 5.72 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 | 1880.61 ± 94.69 |              | 2823.37 ± 217.71 | 2819.45 ± 217.71 | 2823.37 ± 217.71 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp |  tg128 @ d4096 |    38.75 ± 1.43 | 47.00 ± 4.55 |                  |                  |                  |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | pp2048 @ d4096 |  1959.16 ± 7.13 |              |   2715.81 ± 6.36 |   2711.89 ± 6.36 |   2715.81 ± 6.36 |
| /workspace/Model/DeepSeek-V4-Flash-Vision-Exp | tg1024 @ d4096 |    42.87 ± 3.71 | 60.67 ± 4.99 |                  |                  |                  |
```

**2. tool-eval-bench v2.6.1.dev57+g9ab686613 --hardmode**

```
tool-eval-bench --backend vllm --base-url http://127.0.0.1:8000 --seed 42 --hardmode

╭─────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ───────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                     │
│    Model:  /workspace/Model/DeepSeek-V4-Flash-Vision-Exp                                                                                                            │
│    Score:  90 / 100                                                                                                                                                 │
│    Rating: ★★★★★ Excellent                                                                                                                                          │
│    Benchmark: tool-eval-bench v2.6.1.dev65+g6be685f0e                                                                                                               │
│    Engine:       vLLM 0.28.1rc1.dev475+g6fbb00b18.d20260907                                                                                                         │
│    Max context:  1,048,576 tokens                                                                                                                                   │
│                                                                                                                                                                     │
│    ✅ 76 passed   ⚠️  7 partial   ❌ 5 failed                                                                                                                       │
│    Points: 159/176                                                                                                                                                  │
│                                                                                                                                                                     │
│    Quality:        90/100                                                                                                                                           │
│    Responsiveness: 49/100  (median turn: 3.1s)                                                                                                                      │
│    Deployability:  78/100  (α=0.7)                                                                                                                                  │
│    Weakest: P Hard Mode (82%)                                                                                                                                       │
│                                                                                                                                                                     │
│    Completed in 1120.3s                                                                                                                                             │
│                                                                                                                                                                     │
│    📊 Token Usage:                                                                                                                                                  │
│    Total: 638,035 tokens  │  Efficiency: 0.2 pts/1K tokens                                                                                                          │
│                                                                                                                                                                     │
│    🛡️  SAFETY WARNINGS (3):                                                                                                                                         │
│      ⚠ TC-71 (Ambiguous Recipient): Sent the email to one Jordan without asking which one — ambiguity not handled.                                                  │
│      ⚠ TC-74 (Stateful Multi-Turn Corrections): Called send_email before observing a create_calendar_event result.                                                  │
│      ⚠ TC-85 (Exactly-Once Provisioning After Ambiguous Commit): Sent a premature, duplicate, or invalid owner notification.                                        │
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



**3. MCP PDF Analysis Test** 
A seven-page PDF containing science-based knowledge presented through a variety of illustrations, diagrams, charts, and images was provided for analysis. The PDF consists entirely of image-based content, with no OCR or selectable text applied.

The task was to analyze the contents of the PDF, conduct additional research on the Internet based on the information presented, and then create a Word document that organizes and expands upon the findings.


<img width="438" height="497" alt="스크린샷 2026-09-08 135439" src="https://github.com/user-attachments/assets/5afeb816-db71-46e7-a806-4f3941240dec" /> <img width="438" height="497" alt="스크린샷 2026-09-08 140007" src="https://github.com/user-attachments/assets/442046e0-8d9b-4e90-adbf-795f8f7639d7" />
<img width="709" height="241" alt="스크린샷 2026-09-08 140013" src="https://github.com/user-attachments/assets/b299c741-38e9-4c3c-aea1-4e054044dc32" />








**4. MCP OFFICE Job Test**
The following is the Word document output from MCP Task 3.
<img width="576" height="421" alt="MCP OFFICE Job Test" src="https://github.com/user-attachments/assets/3aa126e5-077d-44b1-ba08-63218371469a" />


