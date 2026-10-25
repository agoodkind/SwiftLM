### `mlx-community/gemma-4-26b-a4b-it-8bit` — release b782

Apple M6 · 32 GB · macOS 27.0 · **release b782** (official `SwiftLM-b782-macos-arm64.tar.gz`) · `scripts/profiling/m6_bench.py` · runs=3 (1 at 32K+) · warmup=1 · gen=128 · temperature 0 · medians

`SSD` = `--stream-experts` (b769 and b773 crash in this mode). 32K was not re-run; on the fix build it swapped (see below).

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| SSD | 512 (533) | 202.4 | 2.65 | 9.67 | 6.53 | 0.0 | 73 | ok |
| SSD | 8192 (9546) | 285.7 | 33.5 | 8.49 | 7.11 | 0.0 | 71 | ok |

---

#### Earlier build (before b782)

Apple M6 · 32 GB · runs=3 (long=1) · warmup=1 · gen=128 · temperature 0 · medians

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 | — | — | — | 3.33 | 3.06 | 28 | **MEM_ABORT** swap grew 3.1 GB |
| Vanilla | 2048 | — | — | — | — | — | — | **SKIPPED_AFTER_ABORT**  |
| Vanilla | 8192 | — | — | — | — | — | — | **SKIPPED_AFTER_ABORT**  |
| Vanilla | 32768 | — | — | — | — | — | — | **SKIPPED_AFTER_ABORT**  |
| SSD | 512 (528) | 219.6 | 2.44 | 8.78 | 6.95 | 0.0 | 73 | ok |
| SSD | 2048 (2291) | 193.7 | 11.93 | 7.7 | 7.32 | 0.0 | 69 | ok |
| SSD | 8192 (9543) | 138.5 | 68.94 | 6.92 | 7.58 | 0.0 | 53 | ok |
| SSD | 32768 | — | — | — | 5.8 | 2.09 | 32 | **MEM_ABORT** swap grew 2.1 GB |

#### `--stream-experts` re-measured with SharpAI/mlx-swift-lm#69 and #71 (b769/b773 crash in this mode)

Apple M6 · 32 GB · runs=3 (long=1) · warmup=1 · gen=128 · temperature 0 · medians

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| SSD | 512 (533) | 90.6 | 5.95 | 9.09 | 6.5 | 0.0 | 71 | ok |
| SSD | 8192 (9546) | 145.7 | 65.61 | 8.18 | 7.3 | 0.0 | 59 | ok |
| SSD | 32768 | — | — | — | 5.79 | 2.04 | 31 | **MEM_ABORT** swap grew 2.0 GB |
