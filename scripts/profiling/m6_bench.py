#!/usr/bin/env python3
"""Memory-guarded SwiftLM benchmark harness (built for the 32 GB M6 Mac mini).

For each config it starts one SwiftLM server, then for each context length it
runs a warm-up plus N measured requests and reports medians. It guards memory
on a small-RAM machine:

  * Before each request it records swap usage and free memory as a baseline.
  * While a request runs it polls GPU in-use memory (ioreg), swap and free memory
    every 0.5 s.
  * If swap grows by more than --swap-abort-gb, or free memory drops below
    --min-free-pct, it kills the server, records a MEM_ABORT row, and skips the
    larger contexts for that config.

Every prompt starts with a unique nonce, so SwiftLM's prompt cache can't turn a
repeated run into a cache hit. Each prompt also contains a planted code word,
and the answer must reproduce it.

Example:
  python3 scripts/profiling/m6_bench.py --model mlx-community/Qwen3.8-27B-4bit \
      --config "Vanilla=" --config "TurboKV=--turbo-kv" \
      --contexts 512,2048,8192,32768 --out docs/profiling/m6/qwen38_27b
"""
import argparse
import json
import os
import random
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

SWIFTLM_PATH = ".build/release/SwiftLM"
FILLER = [
    "The harbor master logged {n} vessels before noon and noted calm water.",
    "Sensor {n} in the greenhouse reported stable humidity and mild warmth.",
    "Ledger entry {n}: shipment of copper wire received, inspected, and shelved.",
    "Trail marker {n} points north along the ridge past the old pine grove.",
]


# ── system memory probes ─────────────────────────────────────────────────────

def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else -1


def gpu_in_use_gb():
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                         capture_output=True, text=True, timeout=5).stdout
    m = re.search(r'"In use system memory"=(\d+)', out)
    return int(m.group(1)) / 1024**3 if m else 0.0


class MemWatch:
    """Polls memory while a request runs. Sets .tripped when a guard fires."""

    def __init__(self, swap_abort_gb, min_free_pct, on_trip):
        self.swap0 = swap_used_gb()
        self.swap_abort_gb, self.min_free_pct, self.on_trip = swap_abort_gb, min_free_pct, on_trip
        self.peak_gpu = self.peak_swap_delta = 0.0
        self.min_free = 100
        self.tripped = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        tick = 0
        while not self._stop.is_set():
            self.peak_gpu = max(self.peak_gpu, gpu_in_use_gb())
            self.peak_swap_delta = max(self.peak_swap_delta, swap_used_gb() - self.swap0)
            if tick % 4 == 0:  # memory_pressure is slower; sample every 2 s
                fp = free_pct()
                if fp >= 0:
                    self.min_free = min(self.min_free, fp)
            tick += 1
            if self.tripped is None:
                if self.peak_swap_delta > self.swap_abort_gb:
                    self.tripped = f"swap grew {self.peak_swap_delta:.1f} GB"
                elif 0 <= self.min_free < self.min_free_pct:
                    self.tripped = f"free memory {self.min_free}%"
                if self.tripped:
                    self.on_trip()
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join(timeout=5)


# ── server lifecycle ─────────────────────────────────────────────────────────

def start_server(model, flags, port, ctx_size, log_path):
    cmd = [SWIFTLM_PATH, "--model", model, "--port", str(port), "--ctx-size", str(ctx_size)] + flags
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            return None
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    return None


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Let the Metal heap drain before the next config loads.
    deadline = time.time() + 60
    while time.time() < deadline and gpu_in_use_gb() > 2.0:
        time.sleep(1)


def log_lines_since(log_path, offset):
    with open(log_path) as f:
        f.seek(offset)
        return f.read()


# ── one request ──────────────────────────────────────────────────────────────

def build_prompt(target_tokens, rng):
    nonce = f"run-{rng.getrandbits(48):012x}"
    code = f"{rng.choice(['PELICAN', 'MARLIN', 'OSPREY', 'HERON'])}-{rng.randint(10, 99)}"
    # ~15 tokens per filler line; leave room for the instructions.
    n_lines = max(1, (target_tokens - 80) // 15)
    lines = [rng.choice(FILLER).format(n=i) for i in range(n_lines)]
    lines.insert(rng.randint(0, len(lines)), f"Note: the secret code word is {code}.")
    prompt = (f"[{nonce}]\n" + "\n".join(lines) +
              "\n\nFirst, state the secret code word from the notes above. "
              "Then write a detailed story of at least 300 words about a lighthouse keeper.")
    return prompt, code


def run_request(port, prompt, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    t_first, text = None, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                t_first = t_first or time.time()
                text.append(delta)
    return t0, t_first, time.time(), "".join(text)


def parse_server_stats(log_text):
    pre = re.findall(r"prefill done \| n_tokens=(\d+), t=([0-9.]+)s, ([0-9.]+)t/s", log_text)
    done = re.findall(r"slot done: id 0 \| gen_tokens=(\d+)", log_text)
    return (pre[-1] if pre else None), (int(done[-1]) if done else None)


def is_degenerate(text):
    words = text.split()
    if len(words) < 20:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / len(grams) < 0.5


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", action="append", required=True,
                    help='NAME=FLAGS, e.g. "TurboKV=--turbo-kv" (repeatable)')
    ap.add_argument("--contexts", default="512,2048,8192")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-runs", type=int, default=1, help="runs for contexts >= --long-threshold")
    ap.add_argument("--long-threshold", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--port", type=int, default=5431)
    ap.add_argument("--swap-abort-gb", type=float, default=2.0)
    ap.add_argument("--min-free-pct", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="output path prefix (writes .jsonl and .md)")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",")]
    ctx_size = max(contexts) + args.gen + 1024
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    raw_path, md_path = args.out + ".jsonl", args.out + ".md"
    rows = []

    for spec in args.config:
        name, _, flag_str = spec.partition("=")
        flags = flag_str.split()
        log_path = f"{args.out}.{re.sub(r'[^A-Za-z0-9]+', '_', name)}.server.log"
        print(f"\n=== {name}  flags={flags or '(none)'}  swap={swap_used_gb():.1f}GB free={free_pct()}%")
        proc = start_server(args.model, flags, args.port, ctx_size, log_path)
        if not proc:
            print("  server failed to start; see", log_path)
            rows.append({"config": name, "context": None, "status": "START_FAIL"})
            continue
        with open(log_path) as f:
            load = re.search(r"\(([0-9.]+)GB model", f.read())
        print(f"  loaded ({load.group(1) if load else '?'} GB weights), GPU in-use {gpu_in_use_gb():.1f} GB")

        aborted = False
        for ctx in contexts:
            if aborted:
                rows.append({"config": name, "context": ctx, "status": "SKIPPED_AFTER_ABORT"})
                continue
            n_runs = args.long_runs if ctx >= args.long_threshold else args.runs
            n_warm = args.warmup if ctx == contexts[0] else 0
            for i in range(n_warm + n_runs):
                warm = i < n_warm
                prompt, code = build_prompt(ctx, rng)
                offset = os.path.getsize(log_path)
                watch = MemWatch(args.swap_abort_gb, args.min_free_pct, on_trip=proc.kill)
                row = {"config": name, "context": ctx, "run": i - n_warm, "warmup": warm}
                with watch:
                    try:
                        t0, t_first, t_end, text = run_request(args.port, prompt, args.gen)
                        ok = True
                    except Exception as e:
                        ok, err = False, str(e)
                row.update(peak_gpu_gb=round(watch.peak_gpu, 2), swap_delta_gb=round(watch.peak_swap_delta, 2),
                           min_free_pct=watch.min_free)
                if watch.tripped or not ok:
                    row["status"] = "MEM_ABORT" if watch.tripped else "REQUEST_FAIL"
                    row["reason"] = watch.tripped or err
                    print(f"  ctx={ctx} {row['status']}: {row['reason']}")
                    rows.append(row)
                    aborted = True
                    break
                time.sleep(0.5)  # let the server flush its "slot done" line
                pre, gen_tokens = parse_server_stats(log_lines_since(log_path, offset))
                gen_tokens = gen_tokens or max(1, len(text.split()))
                row.update(
                    status="OK",
                    prompt_tokens=int(pre[0]) if pre else None,
                    prefill_tps=float(pre[2]) if pre else None,
                    ttft_s=round(t_first - t0, 2) if t_first else None,
                    decode_tps=round((gen_tokens - 1) / (t_end - t_first), 2) if t_first and gen_tokens > 1 else None,
                    gen_tokens=gen_tokens,
                    needle_ok=code in text,
                    degenerate=is_degenerate(text),
                    code=code,
                    answer_head=text[:240],  # enough to see what a needle miss actually said
                )
                rows.append(row)
                tag = "warm" if warm else f"run{row['run']}"
                print(f"  ctx={ctx:>6} {tag:<5} prompt={row['prompt_tokens']} prefill={row['prefill_tps']} t/s "
                      f"ttft={row['ttft_s']}s decode={row['decode_tps']} t/s gen={gen_tokens} "
                      f"needle={'ok' if row['needle_ok'] else 'MISS'}{' DEGEN' if row['degenerate'] else ''} "
                      f"| gpu={row['peak_gpu_gb']}GB swapΔ={row['swap_delta_gb']}GB free≥{row['min_free_pct']}%")
                with open(raw_path, "a") as f:
                    f.write(json.dumps({"model": args.model, **row}) + "\n")
        stop_server(proc)

    write_markdown(md_path, args, rows)
    print("\nwrote", md_path, "and", raw_path)


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def write_markdown(path, args, rows):
    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
                        capture_output=True, text=True).stdout.split("\n")
    with open(path, "w") as f:
        f.write(f"### `{args.model}`\n\n")
        f.write(f"{hw[0]} · {int(hw[1]) / 1024**3:.0f} GB · runs={args.runs} (long={args.long_runs}) · "
                f"warmup={args.warmup} · gen={args.gen} · temperature 0 · medians\n\n")
        f.write("| Config | Context (prompt tok) | Prefill tok/s | TTFT s | Decode tok/s | Peak GPU GB | Swap Δ GB | Min free % | Checks |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        keys = []
        for r in rows:
            k = (r["config"], r["context"])
            if k not in keys:
                keys.append(k)
        for cfg, ctx in keys:
            every = [r for r in rows if (r["config"], r["context"]) == (cfg, ctx)]
            # A case that aborted during its warm-up has no measured rows; report the warm-up.
            group = [r for r in every if not r.get("warmup")] or every
            bad = [r for r in group if r.get("status") != "OK"]
            if bad:
                f.write(f"| {cfg} | {ctx} | — | — | — | {bad[0].get('peak_gpu_gb', '—')} | "
                        f"{bad[0].get('swap_delta_gb', '—')} | {bad[0].get('min_free_pct', '—')} | "
                        f"**{bad[0]['status']}** {bad[0].get('reason', '')} |\n")
                continue
            checks = "ok" if all(r["needle_ok"] and not r["degenerate"] for r in group) else \
                f"needle {sum(r['needle_ok'] for r in group)}/{len(group)}, degen {sum(r['degenerate'] for r in group)}"
            f.write(f"| {cfg} | {ctx} ({med([r['prompt_tokens'] for r in group])}) | "
                    f"{med([r['prefill_tps'] for r in group])} | {med([r['ttft_s'] for r in group])} | "
                    f"{med([r['decode_tps'] for r in group])} | {max(r['peak_gpu_gb'] for r in group)} | "
                    f"{max(r['swap_delta_gb'] for r in group)} | {min(r['min_free_pct'] for r in group)} | {checks} |\n")


if __name__ == "__main__":
    main()
