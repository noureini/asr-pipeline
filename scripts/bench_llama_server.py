"""Prove Python-level parallelism via llama-server continuous batching.

Launches the locally-built CUDA llama-server with --parallel N (ONE model,
shared weights, N KV slots), then fires N concurrent fusion requests from
Python threads. The server batches them on the single GPU. Reports
aggregate tok/s vs the ~20 tok/s sequential baseline.

Usage:
  uv run python scripts/bench_llama_server.py \
    --model gemma-3-12b-it-UD-Q4_K_XL.gguf \
    --repo unsloth/gemma-3-12b-it-GGUF --parallel 3 --n 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os
BIN = os.path.expanduser("~/asr-pipeline/vendor/llamacpp/llama-server")
LIBDIR = os.path.expanduser("~/asr-pipeline/vendor/llamacpp")
PH = ("আমার ফ্যামিলিতে আমার ছয় জন মানুষ এই ছয় জন মানুষে আমার এক কৃষি কি "
      "কাজের উপরে নির্ভর করে কুনু বছর যদি মাইর যায় তখনি কষ্ট বেড়ে যায়")
# 3 Whisper candidate readings = the real N-best workload (long prompt).
FL_NBEST = ("  (a) আমার ফ্যামিলিতে ছয় জন মানুষ। এই ছয় জন মানুষ আমার কৃষি "
            "কাজের উপরে নির্ভর করে। কোনো বছর যদি ফসল নষ্ট হয়, কষ্ট বাড়ে।\n"
            "  (b) আমার ফ্যামিলিতে ছয়জন মানুষ, এই ছয়জন আমার কৃষি কাজের "
            "উপর নির্ভর করে, কোন বছর ফসল মাইর গেলে কষ্ট বেড়ে যায়।\n"
            "  (c) আমার ফ্যামিলিতে ছয় জন, কৃষি কাজের উপরে নির্ভর করি, "
            "কোনো বছর মাইর যায় তখনই কষ্ট বাড়ে।")
PORT = 8081


def _wait_health(timeout=180):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(2)
    return False


def _one_request(prompt: str):
    body = json.dumps({"prompt": prompt, "temperature": 0.0,
                       "n_predict": 384}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d.get("content", ""), d.get("tokens_predicted", 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-12b-it-UD-Q4_K_XL.gguf")
    ap.add_argument("--repo", default="unsloth/gemma-3-12b-it-GGUF")
    ap.add_argument("--parallel", type=int, default=3, help="server slots")
    ap.add_argument("--n", type=int, default=12, help="concurrent reqs")
    ap.add_argument("--ctx-per-slot", type=int, default=2048)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from asr_pipeline.fusion import load_prompts
    model_path = hf_hub_download(repo_id=args.repo, filename=args.model,
                                 local_files_only=True)
    tpl = Template(load_prompts("config/prompts/bn_survey.yaml")
                   ["fusion_prompt"])
    prompt = tpl.safe_substitute(phonetic=PH, fluent=FL_NBEST)
    print(f">>> prompt length: {len(prompt)} chars (realistic N-best)")

    if not Path(BIN).exists():
        print(f"llama-server not built yet at {BIN}", file=sys.stderr)
        return 1

    total_ctx = args.parallel * args.ctx_per_slot
    cmd = [BIN, "-m", model_path, "-ngl", "99", "--flash-attn", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-c", str(total_ctx),
           "--parallel", str(args.parallel), "--cont-batching",
           "--host", "127.0.0.1", "--port", str(PORT)]
    print(">>> launching:", " ".join(cmd[3:]))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, env=env)
    try:
        if not _wait_health():
            print("server did not become healthy", file=sys.stderr)
            return 1
        print(f">>> server up. Firing {args.n} concurrent requests "
              f"at {args.parallel} slots...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.n) as ex:
            results = list(ex.map(_one_request, [prompt] * args.n))
        dt = time.perf_counter() - t0
        toks = sum(t for _, t in results)
        print(f"\n=== {args.n} reqs, {args.parallel} slots: {dt:.1f}s wall"
              f" | {toks} tokens | AGGREGATE {toks/dt:.1f} tok/s | "
              f"{dt/args.n:.1f}s/req effective ===")
        print("baseline sequential ~20 tok/s, ~15 s/segment")
        print("sample:", results[0][0][:110].replace("\n", " "))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
