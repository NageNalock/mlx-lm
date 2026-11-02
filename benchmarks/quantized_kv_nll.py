# Copyright © 2026 Apple Inc.
"""Compare teacher-forced code continuation loss with quantized KV caches."""

import argparse
import gc
import json
import runpy
from pathlib import Path

import mlx.core as mx
from quantized_kv_quality import load_cases

from mlx_lm import load
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models import base
from mlx_lm.models.cache import make_prompt_cache


def measure(model, tokens, bits, target_tokens):
    cache = make_prompt_cache(model)
    split = len(tokens) - target_tokens
    for start in range(0, split - 1, 2048):
        end = min(start + 2048, split - 1)
        model(mx.array(tokens[start:end])[None], cache=cache)
        maybe_quantize_kv_cache(cache, 0, 64, bits)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
    logits = model(mx.array(tokens[split - 1 : -1])[None], cache=cache)[0]
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = mx.array(tokens[split:])
    losses = -mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
    correct = (mx.argmax(logits, axis=-1) == targets).sum()
    mx.eval(losses, correct)
    result = {
        "nll": losses.mean().item(),
        "target_tokens": target_tokens,
        "correct": correct.item(),
        "losses": losses.tolist(),
    }
    del cache, logits, logprobs, losses, correct
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=128)
    args = parser.parse_args()
    model, tokenizer = load(args.model, revision=args.revision)
    baseline_file = args.baseline / "mlx_lm/models/base.py"
    original = runpy.run_path(str(baseline_file))[
        "quantized_scaled_dot_product_attention"
    ]
    patched = base.quantized_scaled_dot_product_attention
    manifest = json.loads(args.manifest.read_text())
    cases = [c for c in load_cases(manifest) if c["dataset"] == "lcc"][: args.samples]
    print(
        json.dumps(
            {"model": args.model, "revision": args.revision, "samples": len(cases)}
        ),
        flush=True,
    )
    for case in cases:
        code = case["prompt"].split("\n\n", 1)[1]
        tokens = tokenizer.encode(code)
        for policy in ("full", "baseline8", "patched8"):
            base.quantized_scaled_dot_product_attention = (
                original if policy == "baseline8" else patched
            )
            result = measure(
                model, tokens, None if policy == "full" else 8, args.target_tokens
            )
            print(
                json.dumps(
                    {"id": case["id"], "mode": policy, "tokens": len(tokens), **result}
                ),
                flush=True,
            )
    base.quantized_scaled_dot_product_attention = patched


if __name__ == "__main__":
    main()
