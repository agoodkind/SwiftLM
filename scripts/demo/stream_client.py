#!/usr/bin/env python3
"""Tiny SwiftLM demo client for terminal recordings.

Streams one chat completion and prints the text live. While it runs it samples
the server's physical footprint and system swap, then prints a stats line. The
figures come from this run, not from the benchmark files.

usage: stream_client.py warmup | short
       stream_client.py long <n_filler_lines>
"""
import json, re, subprocess, sys, threading, time, urllib.request

PORT = 5431
DIM, BOLD, CYAN, GREEN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"


def server_pid():
    out = subprocess.run(["pgrep", "-n", "-f", "SwiftLM --model"], capture_output=True, text=True).stdout.split()
    return out[0] if out else None


def footprint_gb(pid):
    out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s+([0-9.]+)\s+(GB|MB)", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v if m.group(2) == "GB" else v / 1024


def swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def main():
    kind = sys.argv[1]
    if kind == "warmup":
        # The first request after load pays one-off Metal pipeline/compile costs.
        prompt = "Count from 1 to 8, comma-separated."
        max_tokens = 24
    elif kind == "short":
        prompt = "Write a Swift function `fib(_ n: Int) -> Int` that returns the n-th Fibonacci number iteratively. Code only, with a one-line doc comment."
        max_tokens = 160
    else:
        n = int(sys.argv[2])
        lines = [f"Sensor {i} in the greenhouse reported stable humidity and mild warmth." for i in range(n)]
        lines.insert(n * 2 // 3, "Note: the secret code word is OSPREY-17.")
        prompt = f"[demo-{int(time.time())}]\n" + "\n".join(lines) + "\n\nWhat is the secret code word? Answer in one short sentence."
        max_tokens = 24

    pid = server_pid()
    swap0, peak_fp, peak_swap = swap_gb(), [0.0], [0.0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak_fp[0] = max(peak_fp[0], footprint_gb(pid))
            peak_swap[0] = max(peak_swap[0], swap_gb() - swap0)
            stop.wait(1)

    threading.Thread(target=sample, daemon=True).start()

    # Client-side prefill ticker. The server's opt-in prefill_progress heartbeat
    # doesn't fire on the VLM path yet, so we show elapsed time and live memory
    # here until the first token arrives.
    first_token = threading.Event()
    out_lock = threading.Lock()  # keeps the ticker from writing over the first token
    t0 = time.time()

    def ticker():
        while not first_token.wait(1):
            with out_lock:
                if first_token.is_set():
                    break
                sys.stdout.write(f"\r{DIM}  prefilling … {time.time() - t0:5.0f}s  | SwiftLM {footprint_gb(pid):4.1f} GB  "
                             f"swap +{max(0, swap_gb() - swap0):.1f} GB{RESET}   ")
                sys.stdout.flush()

    if kind != "short":
        threading.Thread(target=ticker, daemon=True).start()

    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json", "X-SwiftLM-Prefill-Progress": "true"})
    t_first = None; n_tok = 0; event = None; n_prompt = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("event:"):
                event = line[6:].strip(); continue
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if event == "prefill_progress":
                n_prompt = d.get("n_prompt_tokens") or n_prompt
                sys.stdout.write(f"\r{DIM}  prefilling {n_prompt or '?':,} tokens … {d.get('elapsed_seconds', time.time() - t0):5.0f}s  "
                                 f"| SwiftLM {footprint_gb(pid):4.1f} GB  swap +{max(0, swap_gb() - swap0):.1f} GB{RESET}   ")
                sys.stdout.flush(); event = None; continue
            event = None
            if d.get("usage"):
                n_prompt = d["usage"].get("prompt_tokens", n_prompt)
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if t_first is None:
                    with out_lock:
                        t_first = time.time(); first_token.set()
                        sys.stdout.write("\r\033[K" + GREEN)
                n_tok += 1
                sys.stdout.write(delta); sys.stdout.flush()
    t_end = time.time(); stop.set(); first_token.set()
    sys.stdout.write(RESET + "\n\n")
    decode = (n_tok - 1) / (t_end - t_first) if n_tok > 1 else 0
    ttft = t_first - t0
    pf = f" prompt {n_prompt:,} tok · prefill {n_prompt / ttft:.0f} tok/s ·" if n_prompt and kind == "long" else ""
    print(f"{CYAN}{BOLD}  {n_tok} tokens · decode {decode:.1f} tok/s · TTFT {ttft:.1f}s ·{pf.rstrip(' ·')}{RESET}")
    print(f"{CYAN}{BOLD}  peak SwiftLM {peak_fp[0]:.1f} GB · swap +{max(0, peak_swap[0]):.1f} GB{RESET}")


if __name__ == "__main__":
    main()
