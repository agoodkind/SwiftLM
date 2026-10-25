### `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` — release b782

Apple M6 · 32 GB · macOS 27.0 · **release b782** (official `SwiftLM-b782-macos-arm64.tar.gz`) · `scripts/profiling/m6_bench.py` · runs=3 (1 at 32K+) · warmup=1 · gen=128 · temperature 0 · medians

`Vanilla` = all experts on the GPU. `SSD` = `--stream-experts`.

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (548) | 686.9 | 0.8 | 48.35 | 20.2 | 0.0 | 25 | ok |
| Vanilla | 2048 (2346) | 960.5 | 2.48 | 46.92 | 20.4 | 0.0 | 26 | ok |
| Vanilla | 8192 (9808) | 926.1 | 10.68 | 44.84 | 20.54 | 0.0 | 25 | ok |
| Vanilla | 32768 (40830) | 675.6 | 60.68 | 36.79 | 22.07 | 0.0 | 20 | ok |
| SSD | 512 (557) | 263.1 | 2.14 | 14.1 | 5.7 | 0.0 | 71 | ok |
| SSD | 2048 (2332) | 407.9 | 5.76 | 14.13 | 5.74 | 0.0 | 72 | ok |
| SSD | 8192 (9820) | 415.3 | 23.71 | 13.87 | 5.89 | 0.0 | 73 | ok |
| SSD | 32768 (40820) | 346.5 | 118.07 | 13.02 | 6.81 | 0.0 | 68 | needle 0/1, degen 0 |

> **40.8K needle check:** the hidden code word is not recalled reliably at 40.8K tokens in either mode, so treat those rows as throughput only. Across 3 SSD runs and 2 GPU re-runs, every miss had an `OSPREY-NN` code word (answered `OSPREY` without the number, or misspelled `OSPERY`); `MARLIN-60` passed in both modes. On the same `OSPREY-67` prompt GPU and SSD fail alike, so it is the model, not SSD streaming. Every run at ≤9.8K passed. The re-runs are in the JSONL with `"recheck": true`.
