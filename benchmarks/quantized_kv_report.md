# Quantized KV prefill on Apple M1 Pro

This review evaluates a small change to quantized attention in `models/base.py`.
Large Metal prefills dequantize one layer's K/V and call the existing fused
SDPA kernel. The stored cache, quantization parameters, token positions, and
decode computation stay the same. Query scaling keeps the original operation
order and dtype.

This addresses the prefill attention-score allocation reported in
[mlx-lm issue 1587](https://github.com/ml-explore/mlx-lm/issues/1587).
[PR 1929](https://github.com/ml-explore/mlx-lm/pull/1929) discusses a different
query-tiling approach to the same allocation problem. This change reuses the
existing MLX kernel and adds no eager evaluation inside the model.

## Scope

The fast path requires Metal, at least 512 query rows, matching query/value
head dimensions of 64 or 128, and either causal attention or no mask.
Small queries, other head dimensions, array masks, and CPU execution use the
existing implementation. In particular, this does not optimize padded batch
masks, 256-dimensional heads, or single-token decode.

The 512-row threshold is conservative: dequantization can increase temporary
memory for smaller multi-head attention calls. No new user option or dependency
is needed. KV quantization itself remains opt-in through the existing API.

Keeping all tokens does not imply bitwise-identical outputs. Fused attention
changes floating-point reductions, and dequantized K/V are rounded to their
floating-point dtype. The quality checks below compare the old and new paths
with the same quantization settings.

## Environment and method

- Apple M1 Pro, 10 CPU cores, 32 GiB unified memory, connected to power.
- macOS 26.6.2, Python 3.12.15, MLX 0.32.4, Transformers 5.22.1.
- Baseline mlx-lm commit: `736ab0d8c1d0edc79c3d798044fe6904e2fa23be`.
- Implementation commit: `2836d8a` (production change and regression tests).
- Qwen: `mlx-community/Qwen3-4B-4bit`, revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.
- Llama: `mlx-community/Llama-3.2-1B-Instruct-4bit`, revision
  `08231374eeacb049a0eade7922910865b8fce912`.

The model weights are 4-bit in every comparison; "KV bits" describes the
attention cache, not the weights. Each performance configuration runs in a
fresh process, loads the same model revision, warms up once with the measured
shape, and then records three trials. Timings exclude model loading. The
prompt uses deterministic random token IDs with seed 0; EOS stopping is
disabled so each trial generates exactly 256 tokens. This workload measures
performance, not answer quality.

Prefill chunks contain 2048 tokens and the KV group size is 64. Most rows
explicitly use `quantized_kv_start=0`; a separate comparison uses the existing
default of 5000. Even with start 0, mlx-lm quantizes the initial ordinary cache
after the first prefill chunk. Performance runs do not overlap other GPU test
or benchmark workloads.

"Peak memory" is the peak active allocation reported by MLX/Metal, including
model weights. It is not process RSS or total system memory. Physical KV
allocation includes cache padding. The benchmark also records allocation after
each cache is released to check for retained request state.

## Performance

Medians of three measured trials, after warmup. All paired rows use the same
model, KV precision, prompt, and generation length. Start is the existing
`quantized_kv_start` setting. Memory uses GiB (2^30 bytes).

| Model | Input tokens | KV bits | Start | TTFT old → new (s) | Peak old → new (GiB) | Decode old → new (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 8192 | 8 | 0 | 33.48 → 28.80 | 5.167 → 3.641 | 34.26 → 34.28 |
| Qwen3-4B | 16384 | 8 | 0 | 88.99 → 71.62 | 7.828 → 4.039 | 25.74 → 25.83 |
| Qwen3-4B | 8192 | 4 | 0 | 33.37 → 28.52 | 4.868 → 3.295 | 35.60 → 35.76 |
| Qwen3-4B | 8192 | 8 | 5000 | 30.19 → 28.42 | 5.167 → 3.641 | 34.37 → 34.01 |
| Llama-3.2-1B | 8192 | 8 | 0 | 6.81 → 5.52 | 4.106 → 1.703 | 108.90 → 108.61 |

At 16K, Qwen's TTFT decreases 19.5% and peak allocation decreases 48.4%.
Total generation time decreases from 98.90 s to 81.60 s. At 8K with the default
start threshold of 5000, TTFT decreases 5.9%; the larger 14.0% improvement
requires the explicitly measured start-0 configuration.

An unquantized Qwen 8K control measures 28.19 s TTFT, 40.22 decode tok/s, and
3.836 GiB peak. The optimized 8-bit path still has the pre-existing quantized
decode cost. Its physical KV allocation at the first yield is 631.125 MiB,
compared with 1188 MiB for the unquantized cache. The optimization does not
further reduce the stored cache: old and new quantized allocations match.
Allocation after each cache release is constant across repetitions.

[Raw performance trials](results/quantized_kv/performance.jsonl) retain measured
values, arguments, versions, and source hashes; only local import paths are
omitted. These results apply to the measured M1 Pro and the supported shapes.

## Quality and regression tests

The implementation was frozen before selecting 48 independent examples: 16
each from Qasper, HotpotQA, and LCC, with 3043–15779 input tokens. None overlaps
the earlier diagnostic or development examples. These runs use Qwen3-4B-4bit,
8-bit KV, greedy generation, and a fresh prefill for every policy.

| Dataset / metric (0–100) | Unquantized | Old 8-bit | New 8-bit |
|---|---:|---:|---:|
| Qasper / token F1 | 31.29 | 31.29 | 31.79 |
| HotpotQA / token F1 | 55.21 | 55.21 | 55.21 |
| LCC / next-line similarity | 43.67 | 45.90 | 47.55 |

Old and new 8-bit outputs are exactly equal on 42/48 examples. Scores are
equal on 43, higher on two, and lower on three. The lower scores are LCC rows
172, 405, and 455; their similarity changes are −4.27, −4.28, and −1.70 points.
The small Qasper gain is extra reference-word overlap in an incorrect answer,
and most of the LCC gain comes from a single example. These changes are not
evidence that the optimization improves model ability.

The [complete independent outputs](results/quantized_kv/quality_validation.jsonl)
and [per-example changes](results/quantized_kv/quality_summary.json) include
the regressions. This small screen found no large aggregate regression; it
does not establish equivalence across models, tasks, or quantization levels.
Generation quality is checked at 8 bits only, although the performance and
numeric attention tests also cover lower precisions.

The 48 earlier diagnostic examples were also rerun with the final
implementation. These are not held-out examples: the earlier screen prompted
the query-scaling correction. The unchanged full and old 8-bit controls are
reused, and all new 8-bit outputs come from fresh prefills.

| Diagnostic dataset / metric (0–100) | Unquantized | Old 8-bit | New 8-bit |
|---|---:|---:|---:|
| Qasper / token F1 | 28.40 | 26.45 | 29.20 |
| HotpotQA / token F1 | 65.83 | 72.08 | 65.83 |
| LCC / next-line similarity | 43.61 | 40.90 | 41.30 |

Here, 43/48 outputs are exactly equal; scores are equal on 43, higher on four,
and lower on one. **The HotpotQA regression remains unresolved.** On row 120,
the full and old 8-bit paths answer "duck" correctly, while the new path
answers "Redhead." This one failure accounts for the 6.25-point difference in
that 16-example diagnostic subset. The independent HotpotQA subset has equal
scores, but that does not erase this known failure. The [diagnostic outputs](results/quantized_kv/quality_diagnostic.jsonl)
retain it alongside the improvements. This is a useful performance change
with measured quality limits, not a claim of lossless generation.

Teacher-forced code continuation, eight fixed examples and 1024 target tokens
per model. Lower mean negative log-likelihood (NLL) is better.

| Model | Unquantized NLL | Old 8-bit NLL | New 8-bit NLL | Old → new correct tokens |
|---|---:|---:|---:|---:|
| Qwen3-4B | 0.696023 | 0.699689 | 0.698275 | 892 → 891 / 1024 |
| Llama-3.2-1B | 0.854884 | 0.854362 | 0.854948 | 846 → 847 / 1024 |

The changes are small and have opposite signs between models. This is a
fidelity check on these code continuations, not evidence of improved model
ability or a broad perplexity benchmark. [Per-token losses](results/quantized_kv/nll.jsonl)
and the [aggregate values](results/quantized_kv/nll_summary.json) are included.

The final standard suite completes successfully: 406 tests, one skip, compared
with 401 tests and the same skip on the baseline. The additional coverage checks
2/3/4/6/8-bit attention against a dense reference in float32, float16, and
bfloat16; causal prefix offsets; routing away from the large scores allocation;
small-query and array-mask fallback; compiled MHA; and CPU fallback. The routing
regression test fails on the unmodified baseline. Black, isort, and Ruff pass
through the repository's `pre-commit run --all-files` configuration.

The separate two-process model-parallel suite reports the same two pre-existing
errors on both revisions: Muse Spark's `ModelArgs` has no top-level
`vocab_size` in `test_shard` and `test_pipeline`. The launcher returns zero even
when workers report these errors, so the worker logs were checked directly.
This review does not claim that the distributed suite is green.
The [test summary](results/quantized_kv/test_summary.json) records both revisions
and distinguishes successful tests from the known distributed failures.

## Reproduce

Create a baseline checkout beside this checkout, using the same Python
environment and package versions:

```bash
git worktree add ../mlx-lm-baseline 736ab0d8c1d0edc79c3d798044fe6904e2fa23be
python -m pip install -e '../mlx-lm-baseline[test]'
python -m pip install -e '.[test]'
python -m pip install mlx==0.32.4 transformers==5.22.1
```

For the standard tests, prepare fixtures using the repository's
[CI workflow](../.github/workflows/pull_request.yml) and allow missing Hugging
Face assets to download once. The recorded baseline and patched runs then use
the same fixture cache in offline mode.

Run each performance case once against each source tree. The JSON metadata
records the imported source path; verify it before comparing results.

```bash
PYTHONPATH=../mlx-lm-baseline python benchmarks/quantized_kv.py \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --prompt-tokens 8192 --kv-bits 8 --label baseline > baseline.jsonl

PYTHONPATH=. python benchmarks/quantized_kv.py \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --prompt-tokens 8192 --kv-bits 8 --label patched > patched.jsonl
```

Use `--prompt-tokens 16384` for the longer input, `--kv-bits 4` for 4-bit
cache, `--kv-bits 0` for an unquantized control, and
`--quantized-kv-start 5000` for the existing default threshold. Model weights
and tokenizer revisions must remain fixed between comparisons.

The quality harness reconstructs prompts from a pinned LongBench archive and
checks their hashes against the sample manifest. It uses greedy generation,
the model's chat template, disabled Qwen thinking, and fresh caches for every
policy. It alternates policy order and runs a real prefill for each policy;
it does not reuse a full-precision prefix to compare quantized prefills.

QA uses the maximum normalized English token F1 over reference answers. Code
uses `SequenceMatcher(autojunk=False)` on the first nonempty, non-fence output
line. These are paired regression screens, not official LongBench scores.
The separate NLL script measures 128 teacher-forced code continuation tokens
on each of eight fixed code examples, with log-softmax computed in float32.

```bash
python benchmarks/quantized_kv_quality.py --baseline ../mlx-lm-baseline \
  --manifest benchmarks/results/quantized_kv/validation_manifest.json \
  > quality.jsonl

python benchmarks/quantized_kv_nll.py --baseline ../mlx-lm-baseline \
  --manifest benchmarks/results/quantized_kv/validation_manifest.json \
  --model mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 > qwen-nll.jsonl
```

The independent validation manifest contains 16 examples per dataset, selected
with seed 4680 and lengths from 3000 to 16384 chat-template tokens. It excludes
all 48 earlier diagnostic examples and the eight initial Qasper development
examples. Selection uses length and source index, never model scores. The
earlier diagnostic screen motivated matching the original query-scaling order;
it is not a held-out result for the final implementation.

## Contribution notes

The production change is limited to the attention dispatch. Benchmark scripts,
manifests, and raw results are review evidence and can be separated from the
implementation commit for a community contribution.

The implementation and this review material were developed with Codex. This
is a fork review, not an upstream submission. The upstream
[contribution policy](https://github.com/ml-explore/mlx-lm/blob/main/CONTRIBUTING.md)
requires the contributor to understand and take responsibility for the code,
disclose AI use, and write their own community posts and PR description.
