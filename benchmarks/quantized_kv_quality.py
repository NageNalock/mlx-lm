# Copyright © 2026 Apple Inc.
"""Compare quantized prefill quality against a baseline checkout.

This is a paired regression screen, not the official LongBench evaluation.
The manifest fixes the examples before evaluation. Prompt text is reconstructed
from the pinned dataset revision and checked against its recorded SHA-256.
"""

import argparse
import gc
import hashlib
import json
import re
import runpy
import string
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from mlx_lm import load, stream_generate
from mlx_lm.models import base
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


def prompt_for(dataset, row):
    if dataset == "qasper":
        return (
            "Read the article below and answer the question with a short phrase. "
            "Answer yes/no for a yes/no question, or unanswerable when the article "
            "gives no answer. Output only your answer.\n\nArticle:\n"
            + row["context"]
            + "\n\nQuestion:\n"
            + row["input"]
            + "\n\nAnswer:"
        )
    if dataset == "hotpotqa":
        return (
            "Answer the question using the passages below. Output only a short "
            "answer without explanation.\n\n"
            + row["context"]
            + "\n\nQuestion:\n"
            + row["input"]
            + "\n\nAnswer:"
        )
    return (
        "Complete the code below by writing only the next line. Do not repeat "
        "the existing code or add explanations or Markdown fences.\n\n" + row["context"]
    )


def load_cases(manifest):
    path = hf_hub_download(
        "zai-org/LongBench",
        "data.zip",
        repo_type="dataset",
        revision=manifest["dataset_revision"],
    )
    with zipfile.ZipFile(path) as archive:
        data = {}
        for dataset in ("qasper", "hotpotqa", "lcc"):
            name = next(
                n for n in archive.namelist() if n.endswith(f"/{dataset}.jsonl")
            )
            data[dataset] = [json.loads(s) for s in archive.read(name).splitlines()]
    for sample in manifest["samples"]:
        dataset, index = sample["id"].split("-")
        row = data[dataset][int(index)]
        prompt = prompt_for(dataset, row)
        assert row["_id"] == sample["source_id"]
        assert hashlib.sha256(prompt.encode()).hexdigest() == sample["prompt_sha256"]
        yield {
            **sample,
            "dataset": dataset,
            "prompt": prompt,
            "answers": row["answers"],
            "max_tokens": 64 if dataset == "lcc" else 128,
        }


def score(text, case):
    if case["dataset"] == "lcc":
        line = next(
            (
                s.strip()
                for s in text.splitlines()
                if s.strip() and not s.strip().startswith("```")
            ),
            "",
        )
        return max(
            SequenceMatcher(None, line, a.strip(), autojunk=False).ratio()
            for a in case["answers"]
        )

    def normalize(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = normalize(text)
    scores = []
    for answer in case["answers"]:
        expected = normalize(answer)
        common = sum((Counter(predicted) & Counter(expected)).values())
        scores.append(
            2 * common / (len(predicted) + len(expected))
            if predicted or expected
            else 1.0
        )
    return max(scores)


def generate(model, tokenizer, tokens, bits, max_tokens):
    cache = make_prompt_cache(model)
    text = []
    first = None
    for response in stream_generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0),
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_bits=bits,
        quantized_kv_start=0,
    ):
        if first is None:
            first = np.array(response.logprobs.astype(mx.float32)).astype(np.float64)
            first -= np.logaddexp.reduce(first)
        text.append(response.text)
    result = {
        "text": "".join(text),
        "generation_tokens": response.generation_tokens,
        "kv_bytes": sum(c.nbytes for c in cache),
        "finish_reason": response.finish_reason,
    }
    del cache, response
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return result, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument(
        "--revision", default="4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    model, tokenizer = load(args.model, revision=args.revision)
    baseline_file = args.baseline / "mlx_lm/models/base.py"
    original = runpy.run_path(str(baseline_file))[
        "quantized_scaled_dot_product_attention"
    ]
    patched = base.quantized_scaled_dot_product_attention
    print(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "manifest_sha256": hashlib.sha256(
                    args.manifest.read_bytes()
                ).hexdigest(),
                "baseline_sha256": hashlib.sha256(
                    baseline_file.read_bytes()
                ).hexdigest(),
                "patched_sha256": hashlib.sha256(
                    Path(base.__file__).read_bytes()
                ).hexdigest(),
                "prefill_step_size": 2048,
                "quantized_kv_start": 0,
            }
        ),
        flush=True,
    )
    policies = ["full", "baseline8", "patched8"]
    for index, case in enumerate(load_cases(manifest)):
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert len(tokens) == case["tokens"]
        logprobs = {}
        for policy in policies[index % 3 :] + policies[: index % 3]:
            base.quantized_scaled_dot_product_attention = (
                original if policy == "baseline8" else patched
            )
            result, logprobs[policy] = generate(
                model,
                tokenizer,
                tokens,
                None if policy == "full" else 8,
                case["max_tokens"],
            )
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "dataset": case["dataset"],
                        "mode": policy,
                        "prompt_tokens": len(tokens),
                        "score": score(result["text"], case),
                        **result,
                    }
                ),
                flush=True,
            )
        a, b = logprobs["baseline8"], logprobs["patched8"]
        print(
            json.dumps(
                {
                    "pair": case["id"],
                    "first_token_agrees": bool(a.argmax() == b.argmax()),
                    "first_token_symmetric_kl": float(
                        0.5 * np.sum((np.exp(a) - np.exp(b)) * (a - b))
                    ),
                }
            ),
            flush=True,
        )
    base.quantized_scaled_dot_product_attention = patched


if __name__ == "__main__":
    main()
