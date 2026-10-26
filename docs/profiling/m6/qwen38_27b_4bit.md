### `mlx-community/Qwen3.8-27B-4bit` — release b782

Apple M6 · 32 GB · macOS 27.0 · **release b782** (official `SwiftLM-b782-macos-arm64.tar.gz`) · `scripts/profiling/m6_bench.py` · runs=3 (1 at 32K+) · warmup=1 · gen=128 · temperature 0 · medians

Only 548 and 9.8K tokens were re-run on b782. Process RAM at 9.8K was 16.1–16.7 GB (server log).

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (548) | 234.3 | 2.38 | 9.51 | 15.37 | 0.0 | 41 | ok |
| Vanilla | 8192 (9815) | 264.4 | 37.21 | 9.16 | 16.08 | 0.0 | 39 | ok |

---

#### Earlier build (before b782)

> **Correction (Sep 24):** the `TurboKV` rows below were run with `--ctx-size`, which on mlx-swift-lm 460ff81 gives the attention layers a `RotatingKVCache`, and `--turbo-kv` only applies to `KVCacheSimple`. So these rows are effectively vanilla, not TurboKV. See #175 for real TurboKV behaviour.

Apple M6 · 32 GB · runs=3 (long=1) · warmup=1 · gen=128 · temperature 0 · medians

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (548) | 233.1 | 2.37 | 9.27 | 15.22 | 0.0 | 43 | ok |
| Vanilla | 2048 (2346) | 241.8 | 9.75 | 9.13 | 16.08 | 0.0 | 44 | ok |
| Vanilla | 8192 (9809) | 237.8 | 41.32 | 8.88 | 16.86 | 0.0 | 43 | ok |
| Vanilla | 32768 (40829) | 200.1 | 204.28 | 7.86 | 18.43 | 0.0 | 38 | ok |
| TurboKV | 512 (548) | 231.4 | 2.37 | 9.29 | 15.16 | 0.0 | 42 | ok |
| TurboKV | 2048 (2356) | 261.2 | 8.93 | 9.19 | 15.5 | 0.0 | 42 | ok |
| TurboKV | 8192 (9811) | 250.4 | 39.25 | 8.89 | 15.9 | 0.0 | 41 | ok |
| TurboKV | 32768 (40806) | 201.7 | 202.54 | 7.87 | 16.86 | 0.0 | 37 | ok |
