"""llama.cpp server wrapper for PARALLEL fusion (continuous batching).

llama-cpp-python decodes one prompt at a time. The C++ llama-server with
--parallel N --cont-batching serves N requests concurrently from ONE
model (shared weights, N KV slots) on a single GPU. This wrapper launches
it and fires many requests concurrently from Python threads — the server
batches them on the GPU. Measured ~4.5x on the 12B N-best fusion workload.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger("asr_pipeline")

_VENDOR = str(Path(__file__).resolve().parents[2] / "vendor" / "llamacpp")


class LlamaServer:
    def __init__(self, model_path: str, n_parallel: int = 3,
                 n_ctx_per_slot: int = 2048, flash_attn: bool = True,
                 kv_quant: bool = True, port: int = 8081,
                 vendor_dir: str = _VENDOR) -> None:
        self._model = model_path
        self._np = max(1, n_parallel)
        self._ctx = n_ctx_per_slot
        self._flash = flash_attn
        self._kvq = kv_quant
        self._port = port
        self._vendor = vendor_dir
        self._bin = os.path.join(vendor_dir, "llama-server")
        self._proc: Optional[subprocess.Popen] = None

    @property
    def available(self) -> bool:
        return os.path.exists(self._bin)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._port}{path}"

    def start(self, timeout: int = 240) -> bool:
        if not self.available:
            logger.warning("  llama-server binary not found at %s",
                           self._bin)
            return False
        cmd = [self._bin, "-m", self._model, "-ngl", "99",
               "--flash-attn", "on" if self._flash else "off",
               "-c", str(self._np * self._ctx),
               "--parallel", str(self._np), "--cont-batching",
               "--host", "127.0.0.1", "--port", str(self._port)]
        if self._kvq:
            cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (self._vendor + ":"
                                  + env.get("LD_LIBRARY_PATH", ""))
        logger.info("  starting llama-server: %d slots x %d ctx "
                    "(flash=%s kv_q8=%s)", self._np, self._ctx,
                    self._flash, self._kvq)
        logf = os.environ.get("LLAMA_SERVER_LOG")
        err = open(logf, "w") if logf else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=err, env=env)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._proc.poll() is not None:
                logger.warning("  llama-server exited during startup")
                return False
            try:
                with urllib.request.urlopen(self._url("/health"),
                                            timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        logger.info("  ✓ llama-server ready (%.0fs)",
                                    time.perf_counter() - t0)
                        return True
            except Exception:
                time.sleep(2)
        logger.warning("  llama-server did not become healthy")
        return False

    def _one(self, prompt: str, max_tokens: int, temperature: float,
             top_p: float) -> str:
        body = json.dumps({
            "prompt": prompt, "temperature": temperature, "top_p": top_p,
            "n_predict": max_tokens, "cache_prompt": True,
        }).encode()
        req = urllib.request.Request(
            self._url("/completion"), data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r).get("content", "") or ""
        except Exception as ex:
            logger.warning("  server request failed: %s",
                           type(ex).__name__)
            return ""

    def generate_batch(self, prompts: list[str], max_tokens: int,
                       temperature: float = 0.0, top_p: float = 1.0,
                       progress_every: int = 25) -> list[str]:
        """Fire all prompts concurrently; the server batches across its
        slots. Returns results in the SAME order as `prompts`."""
        results: list[str] = [""] * len(prompts)
        done = 0
        # A few more workers than slots so a slot never idles between reqs.
        workers = min(len(prompts), self._np * 2)

        def task(i: int) -> None:
            nonlocal done
            results[i] = self._one(prompts[i], max_tokens, temperature,
                                   top_p)
            done += 1
            if done % progress_every == 0 or done == len(prompts):
                logger.info("    server batched %d/%d", done, len(prompts))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(task, range(len(prompts))))
        return results

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
