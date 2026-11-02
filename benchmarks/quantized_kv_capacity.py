# Copyright © 2026 Apple Inc.
"""Measure long-context capacity in a fresh macOS process per configuration.

The memory budget is checked after each prefill chunk. Exceeding it is a
budget failure, not an out-of-memory result. System swap counters include
other applications and must not be attributed solely to this process.
"""

import argparse
import gc
import hashlib
import json
import platform
import re
import resource
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from mlx_lm import load, stream_generate
from mlx_lm.models import base
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


def system_state():
    vm = subprocess.check_output(["vm_stat"], text=True)
    fields = {
        name: int(value)
        for name, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.M)
    }
    page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    status = subprocess.check_output(
        ["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"], text=True
    )
    return {
        "swap_used_bytes": int(
            float(re.search(r"used = ([\d.]+)M", status).group(1)) * 2**20
        ),
        "swapin_bytes": fields["Swapins"] * page_size,
        "swapout_bytes": fields["Swapouts"] * page_size,
        "compressed_bytes": fields["Pages occupied by compressor"] * page_size,
        "pressure_level": int(status.strip().rsplit(":", 1)[1]),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def emit(record):
    print(json.dumps(record), flush=True)


class BudgetExceeded(Exception):
    pass


def measure(model, tokenizer, tokens, args, budget, phase):
    cache = make_prompt_cache(model)
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    system_before = system_state()
    started = time.perf_counter()
    first = None
    generated = []
    last_report = -2048
    processed_tokens = 0

    def progress(done, total):
        nonlocal last_report, processed_tokens
        processed_tokens = done
        peak = mx.get_peak_memory()
        if peak > budget:
            raise BudgetExceeded(f"MLX peak {peak} exceeds budget {budget}")
        if (
            args.max_prefill_seconds
            and time.perf_counter() - started > args.max_prefill_seconds
        ):
            raise BudgetExceeded(f"Prefill exceeded {args.max_prefill_seconds} seconds")
        if done - last_report >= 2048 or done == total:
            state = system_state()
            emit(
                {
                    "event": "prefill",
                    "phase": phase,
                    "processed": done,
                    "total": total,
                    "elapsed_s": time.perf_counter() - started,
                    "active_bytes": mx.get_active_memory(),
                    "peak_bytes": peak,
                    "system": state,
                }
            )
            last_report = done
            if state["pressure_level"] >= 4:
                raise BudgetExceeded("System memory pressure is critical")

    status = "completed"
    reason = None
    response = None
    try:
        for response in stream_generate(
            model,
            tokenizer,
            tokens,
            max_tokens=args.generation_tokens if phase != "warmup" else 8,
            sampler=make_sampler(temp=0),
            prompt_cache=cache,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits or None,
            kv_group_size=64,
            quantized_kv_start=args.quantized_kv_start,
            prompt_progress_callback=progress,
        ):
            if first is None:
                first = time.perf_counter()
                kv_bytes_first = sum(c.nbytes for c in cache)
                emit(
                    {
                        "event": "first_token",
                        "phase": phase,
                        "ttft_s": first - started,
                        "kv_bytes": kv_bytes_first,
                        "peak_bytes": mx.get_peak_memory(),
                    }
                )
            generated.append(response.token)
    except BudgetExceeded as error:
        status, reason = "stopped", str(error)
    elapsed = time.perf_counter() - started
    peak = mx.get_peak_memory()
    result = {
        "event": "result",
        "phase": phase,
        "status": status,
        "reason": reason,
        "within_mlx_budget": peak <= budget,
        "prompt_tokens": len(tokens),
        "processed_prompt_tokens": processed_tokens,
        "generation_tokens": len(generated),
        "ttft_s": first - started if first is not None else None,
        "decode_tps": (
            (len(generated) - 1) / (elapsed - (first - started))
            if len(generated) > 1
            else None
        ),
        "total_s": elapsed,
        "peak_bytes": peak,
        "kv_bytes_end": sum(c.nbytes for c in cache),
        "system_before": system_before,
        "system_after": system_state(),
        "output_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
    }
    del cache, response
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    result["active_bytes_after_release"] = mx.get_active_memory()
    emit(result)
    return status == "completed" and peak <= budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4, 8), default=8)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=512)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--max-prefill-seconds", type=float, default=0)
    args = parser.parse_args()
    info = mx.device_info()
    budget = (
        int(args.budget_gib * 2**30)
        if args.budget_gib is not None
        else info["max_recommended_working_set_size"]
    )
    emit(
        {
            "event": "metadata",
            "arguments": vars(args),
            "mlx_budget_bytes": budget,
            "budget_is_hardware_limit": False,
            "device": {
                k: info[k]
                for k in (
                    "device_name",
                    "memory_size",
                    "max_recommended_working_set_size",
                )
            },
            "platform": platform.platform(),
            "mlx": version("mlx"),
            "transformers": version("transformers"),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(
                Path(base.__file__).read_bytes()
            ).hexdigest(),
            "system": system_state(),
        }
    )
    started = time.perf_counter()
    model, tokenizer, config = load(
        args.model, revision=args.revision, return_config=True
    )
    emit(
        {
            "event": "loaded",
            "elapsed_s": time.perf_counter() - started,
            "active_bytes": mx.get_active_memory(),
            "peak_bytes": mx.get_peak_memory(),
            "system": system_state(),
        }
    )
    tokenizer._eos_token_ids = set()
    mx.random.seed(0)
    tokens = mx.random.randint(0, config["vocab_size"], (args.prompt_tokens,)).tolist()
    if args.warmup_tokens and not measure(
        model, tokenizer, tokens[: args.warmup_tokens], args, budget, "warmup"
    ):
        return
    for trial in range(args.trials):
        if not measure(model, tokenizer, tokens, args, budget, f"trial-{trial}"):
            break


if __name__ == "__main__":
    main()
