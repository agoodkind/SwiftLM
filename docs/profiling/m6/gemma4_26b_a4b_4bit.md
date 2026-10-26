### `mlx-community/gemma-4-26b-a4b-it-4bit` — release b782

Apple M6 · 32 GB · macOS 27.0 · **release b782** (official `SwiftLM-b782-macos-arm64.tar.gz`) · `scripts/profiling/m6_bench.py` · runs=3 (1 at 32K+) · warmup=1 · gen=128 · temperature 0 · medians

Only the short prompt was re-run on b782.

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (533) | 780.9 | 0.73 | 53.86 | 14.29 | 0.0 | 46 | ok |

---

#### Earlier build (before b782)

> **Correction (Sep 24):** the `TurboKV` rows below were run with `--ctx-size`, which on mlx-swift-lm 460ff81 gives the attention layers a `RotatingKVCache`, and `--turbo-kv` only applies to `KVCacheSimple`. So these rows are effectively vanilla, not TurboKV. See #175 for real TurboKV behaviour.

Apple M6 · 32 GB · runs=3 (long=1) · warmup=1 · gen=128 · temperature 0 · medians

| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |
|---|---|---|---|---|---|---|---|---|
| Vanilla | 512 (533) | 733.3 | 0.77 | 52.24 | 14.48 | 0.0 | 43 | ok |
| Vanilla | 2048 (2284) | 962.9 | 2.45 | 50.24 | 14.95 | 0.0 | 40 | ok |
| Vanilla | 8192 (9543) | 958.8 | 10.14 | 45.09 | 15.78 | 0.0 | 39 | ok |
| Vanilla | 32768 (39772) | 756.8 | 53.33 | 31.03 | 17.91 | 0.0 | 32 | ok |
| Vanilla | 65536 (80710) | 621.7 | 131.31 | 24.34 | 19.48 | 0.0 | 25 | ok |
| TurboKV | 512 (531) | 784.5 | 0.72 | 53.17 | 14.37 | 0.0 | 43 | ok |
| TurboKV | 2048 (2298) | 968.5 | 2.44 | 50.46 | 14.93 | 0.0 | 40 | ok |
| TurboKV | 8192 (9571) | 970.6 | 10.05 | 45.52 | 15.88 | 0.0 | 39 | ok |
| TurboKV | 32768 (39672) | 786.6 | 51.13 | 31.47 | 17.55 | 0.0 | 32 | ok |
| TurboKV | 65536 (80922) | 630.3 | 129.75 | 24.51 | 19.28 | 0.0 | 26 | ok |
| MTP | None | — | — | — | — | — | — | **START_FAIL**  |
