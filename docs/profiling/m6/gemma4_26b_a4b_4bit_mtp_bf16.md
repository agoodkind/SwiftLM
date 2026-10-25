### `mlx-community/gemma-4-26b-a4b-it-4bit`

> Measured on an earlier build (before b782).

Apple M6 · 32 GB · runs=3 (long=1) · warmup=1 · gen=128 · temperature 0 · medians

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (533) | 788.5 | 0.72 | 52.95 | 14.5 | 0.0 | 45 | ok |
| Vanilla | 2048 (2284) | 974.8 | 2.41 | 50.76 | 15.14 | 0.0 | 41 | ok |
| Vanilla | 8192 (9543) | 972.3 | 10.01 | 45.96 | 16.71 | 0.0 | 41 | ok |
| MTP-bf16 | 512 (522) | 791.9 | 0.71 | 45.18 | 16.33 | 0.0 | 40 | ok |
| MTP-bf16 | 2048 (2286) | 973.5 | 2.4 | 35.63 | 16.3 | 0.0 | 42 | ok |
| MTP-bf16 | 8192 (9552) | 955.0 | 10.17 | 30.35 | 16.66 | 0.0 | 38 | ok |
