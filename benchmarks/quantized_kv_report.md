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

At 16K with these 2048-token chunks, Qwen's TTFT decreases 19.5% and peak
allocation decreases 48.4%. This is not a comparison against the baseline's
minimum attainable memory use; the capacity screen below also tunes chunks.
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

## Larger-model capacity screen

The follow-up asks whether this change enables a larger model on the same
32 GiB Mac for long-document or code analysis. Before measuring the 14B and
32B models, the [protocol](results/quantized_kv/capacity_protocol.json) set a
provisional limit of 300 seconds to the first token and at least 5 generated
tokens/s. The user accepts several minutes of initial processing; 300 seconds
is an explicit experimental choice, not an exact user requirement.

The MLX allocation budget is the device's recommended working set:
26,800,603,136 bytes (24.960 GiB). It is neither a physical memory limit nor
a simulation of a smaller Mac. A budget stop is not an OOM. The process also
stops after a prefill chunk exceeds its time limit, or under critical
system memory pressure. Such runs have no completed TTFT; their elapsed time
is a lower bound. A watchdog checks pressure during model loading as well.

An [addendum](results/quantized_kv/capacity_protocol_addendum.json), recorded
before any 32B measurement, allows the key 32B 16K comparisons to complete
up to a 900-second prefill ceiling. This measures subsequent generation even
if the first token takes longer than the provisional limit. The pass criteria
remain 300 seconds and 5 tokens/s; a slower completion is not reclassified
as a pass. Memory and pressure limits are unchanged.

Each configuration uses a fresh process, a 512-token warmup, and one measured
trial with 256 forced output tokens. This differs from the full-shape warmup
and three-trial timing comparisons above. These are capacity screens, not
precise speedup estimates. Latency excludes model loading and warmup; loading
time and allocation are retained in the raw events. Random token IDs measure
allocation and execution; they do not establish long-document or code-analysis
answer quality.

All weights remain 4-bit. The primary comparison keeps 8-bit KV, group size
64, and quantization start 0. The baseline may reduce its prefill chunk size,
and unquantized fused attention is also evaluated where it fits. Existing
KV quantization savings are not a benefit introduced by this patch.

A preliminary Qwen3-4B 16K control demonstrates why chunk tuning matters.
Original 8-bit KV with chunks of 256 peaks at 4.498 GiB; with chunks of 128,
it peaks at 3.955 GiB. Both complete below 90 seconds to the first token.
The new path with 2048-token chunks takes 72.76 seconds and peaks at
4.039 GiB using the same capacity harness. Thus, the 48.4% default-chunk
reduction does not demonstrate a lower memory floor than a tuned baseline.
These single-trial screens support a speed/memory tradeoff, not a precise
speedup estimate against every possible baseline configuration.

Qwen3-14B-4bit, revision `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`,
with 16384 input tokens:

| Path | KV bits | Chunk | TTFT (s) | Decode (tok/s) | MLX peak (GiB) | Process footprint peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Original | Unquantized | 2048 | 193.12 | 12.32 | 10.804 | 13.274 |
| Original | 8 | 2048 | 216.92 | 12.29 | 14.483 | 15.977 |
| Original | 8 | 128 | 218.19 | 11.54 | 9.693 | 11.151 |
| New | 8 | 2048 | 194.36 | 12.22 | 9.874 | 11.276 |

All four configurations meet the provisional performance limits. The
default-chunk improvement is useful, but the baseline already runs this
model and context. Its 128-token chunks use slightly less peak MLX memory
than the new 2048-token path. The unquantized baseline has similar timing
to the new quantized path, at a larger cache size.

An additional new-path 14B 32768-token screen stops after processing 24576
tokens in 324.26 seconds. Its peak allocation at that point is 10.482 GiB,
below the memory budget, and it produces no output tokens. This is a latency
screen failure, not an OOM or a completed 32K memory measurement. It shows
that memory savings alone do not meet the chosen waiting-time target.

Qwen3-32B-4bit, revision `bcaaf7f538adf166c1080a2befdb4f6019f66639`,
is measured after both model downloads finish. The weight files were checked
against their official Hugging Face SHA-256 metadata; the
[asset manifest](results/quantized_kv/capacity_model_assets.json) records each
file. Weights alone occupy 17.164 GiB of MLX allocation.

| Path | Input tokens | KV bits | Chunk | TTFT (s) | Decode (tok/s) | MLX peak (GiB) | Process footprint peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8192 | Unquantized | 2048 | 206.67 | 6.85 | 19.957 | 21.651 |
| Original | 16384 | 8 | 128 | 525.29 | 5.35 | 20.164 | 22.409 |
| New | 16384 | 8 | 2048 | 465.73 | 5.27 | 20.035 | 22.333 |

The original code completes the 32B 8K request inside the latency and decode
limits. However, during this first 32B load/warmup the system briefly reports
critical pressure and swap use rises by 4.644 GiB. The measured request then
runs at normal pressure with no further swap-out increase. This row therefore
passes the timing/allocation checks but not the combined pressure screen.
Later processes start with those pages already swapped out. Their smaller
swap deltas cannot be attributed to this patch; model loading is unchanged.

Both 16K paths complete 256 output tokens, with normal pressure throughout
their measured requests. Both fail the predeclared 300-second TTFT screen,
while allocation and subsequent throughput meet their limits. The new path
takes about 60 seconds less to the first token in this screen; the two
paths have similar peak memory and decode throughput. Their stored KV
allocations are identical: 2,317,352,960 bytes. The small throughput difference
does not demonstrate a decode change; production decode code is unchanged.

**These measurements do not establish a larger-model capability gain.**
With a 300-second initial-wait limit, neither 32B 16K configuration qualifies.
With a more relaxed limit, both run. The original already runs 14B at 16K and
32B at 8K. This patch is a prefill optimization, not a reduction in model
weights or permanent KV storage. The results do not support a claim that it
upgrades this Mac from 14B-only to 32B-capable.

The 32B comparison is a single screen of two chunk choices at fixed KV
precision. Unquantized 32B 16K, lower KV precisions, other model families,
and a 32B 32K run remain unevaluated. The 14B and 32B model outputs here
have no task-quality assessment; the separate quality checks below use
smaller models.

MLX peak includes model load, warmup, and the measured trial. The separate
process footprint comes from macOS `/usr/bin/time -l`; it includes memory
that MLX's active allocation counter does not. System-wide swap and pressure
also include other applications. Downloads overlapped the initial 4B and
14B screens, so these runs are not clean estimates of small timing changes.
No GPU benchmark jobs overlap.

[Raw capacity events](results/quantized_kv/capacity.jsonl) retain progress,
stopped runs, source hashes, and process measurements. The
[summary](results/quantized_kv/capacity_summary.json) applies the performance
limits consistently. It separates latency/throughput, allocation, and observed
critical-pressure flags. Its combined screen requires all three to pass,
including model loading and warmup. Passing this synthetic screen alone does
not establish model quality or a general claim of practical usability.

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

For the capacity screen, download the pinned model once, then run each
configuration in a separate process. Keep the same device, weight revision,
input length, and generation length. The macOS `time` output records process
footprint separately from MLX allocation. For the 32B 16K completion comparison:

```bash
PYTHONPATH=../mlx-lm-baseline /usr/bin/time -l python benchmarks/quantized_kv_capacity.py \
  --model mlx-community/Qwen3-32B-4bit \
  --revision bcaaf7f538adf166c1080a2befdb4f6019f66639 \
  --label baseline-32b-16k --prompt-tokens 16384 --kv-bits 8 \
  --prefill-step-size 128 --max-prefill-seconds 900 \
  > baseline-capacity.jsonl 2> baseline-capacity.time

PYTHONPATH=. /usr/bin/time -l python benchmarks/quantized_kv_capacity.py \
  --model mlx-community/Qwen3-32B-4bit \
  --revision bcaaf7f538adf166c1080a2befdb4f6019f66639 \
  --label patched-32b-16k --prompt-tokens 16384 --kv-bits 8 \
  --prefill-step-size 2048 --max-prefill-seconds 900 \
  > patched-capacity.jsonl 2> patched-capacity.time
```

Use `--max-prefill-seconds 300` for an early latency screen. A stopped run
records its reason and completed prompt-token count, with no completed TTFT.
The default allocation budget comes from the device's recommended working set.
The recorded runs also use an external watchdog with a 30-minute process
ceiling, stopping after three consecutive critical-pressure readings sampled
five seconds apart. This watchdog only terminates its own process group.

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
