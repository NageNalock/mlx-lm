# Copyright © 2026 Apple Inc.
"""Measure quantized KV generation in a fresh process for each revision.

Example:
    python benchmarks/quantized_kv.py --model mlx-community/Qwen3-4B-4bit \
        --prompt-tokens 8192 --kv-bits 8 > results.jsonl

Use --kv-bits 0 for an unquantized control. Model loading and one full warmup
are excluded. Output is JSONL; timings include Python generation overhead.
"""

import argparse
import gc
import hashlib
import json
import platform
import time
from importlib.metadata import version

import mlx.core as mx

import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


def measure(model, tokenizer, tokens, args):
    cache = make_prompt_cache(model)
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    first = None
    generated = []
    for response in stream_generate(
        model,
        tokenizer,
        tokens,
        max_tokens=args.generation_tokens,
        sampler=make_sampler(temp=0),
        prompt_cache=cache,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits or None,
        kv_group_size=64,
        quantized_kv_start=args.quantized_kv_start,
    ):
        if first is None:
            first = time.perf_counter()
            kv_bytes_first = sum(c.nbytes for c in cache)
        generated.append(response.token)
    elapsed = time.perf_counter() - start
    ttft = first - start
    result = {
        "prompt_tokens": len(tokens),
        "generation_tokens": response.generation_tokens,
        "ttft_s": ttft,
        "decode_tps": (response.generation_tokens - 1) / (elapsed - ttft),
        "total_s": elapsed,
        "peak_bytes": mx.get_peak_memory(),
        "kv_bytes_first": kv_bytes_first,
        "kv_bytes_end": sum(c.nbytes for c in cache),
        "output_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
    }
    del cache, response
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    result["active_bytes_after_release"] = mx.get_active_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4, 8), default=8)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    model, tokenizer, config = load(
        args.model, revision=args.revision, return_config=True
    )
    tokenizer._eos_token_ids = set()
    mx.random.seed(0)
    tokens = mx.random.randint(0, config["vocab_size"], (args.prompt_tokens,)).tolist()
    print(
        json.dumps(
            {
                "environment": {
                    "platform": platform.platform(),
                    "mlx": version("mlx"),
                    "mlx_lm_source": mlx_lm.__file__,
                    "transformers": version("transformers"),
                },
                "arguments": vars(args),
            }
        ),
        flush=True,
    )
    measure(model, tokenizer, tokens, args)
    for trial in range(args.trials):
        print(
            json.dumps({"trial": trial, **measure(model, tokenizer, tokens, args)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
