# TurboQuant decode on M1 Pro

The default route skips P256 for all eight head-dimension-256 cases at 4K
tokens with one request. Forcing P256 cut attention time by **39–64%**
using the median of the three time ratios for each case. All three
sweeps showed a reduction.

Across 2K–16K tokens, the default route skipped 24 of 64 head-dimension-256
cases. P256 was at least 5% faster in every sweep for 21 of the 24. The
other three were 2K single-request cases with 24 query heads. Four
head-dimension-128 cases below the cutoff were at least 5% slower with
P256 in every sweep.

## Where the default route changes

P256 requires at least 528 units of work on the 16-core M1 Pro. Each
request contributes one unit per query head for every full 256-token
partition. The table gives the first eligible length per request when all
requests have equal lengths. Both page sizes and both dtypes switched at
the listed boundary.

| Query heads / KV heads | One request | Four requests |
| --- | ---: | ---: |
| 24 / 4 | 5,632 | 1,536 |
| 16 / 2 | 8,448 | 2,304 |

The default native call selected the expected route on all 152 checks,
including the token just below each boundary. Its output matched the
corresponding forced or disabled path exactly on those checks. At 3,072
tokens with one request, P256 reduced median attention time by **47–63%**
across the eight head-dimension-256 cases. It was also faster in all repeats
one token below 5,632 and 8,448.

Across the 336 sweep measurements, the largest absolute output difference
from the established kernel was 0.0001221 for FP16 and 0.0009766 for BF16.
Six existing TurboQuant routing tests passed on the same build.

## Setup

Apple M1 Pro with 16 GPU cores and 32 GiB, macOS 26.6.2 (25G83), on AC power.
The native code came from vLLM Metal `0.33.0.dev20261112090227` at
`3451a0149531e4d2889488ed1a80b5a74f6bb2c5`, with MLX 0.32.4,
vLLM 0.33.0+cpu, Python 3.12.16, and NumPy 2.3.5.
[Release wheel](https://github.com/vllm-project/vllm-metal/releases/tag/v0.33.0.dev20261112090227)

The benchmark uses synthetic attention inputs with Q8_0 keys and 3-bit values.
It feeds the same inputs to the established kernel and to GQA with 256-token
and 512-token partitions. Each sweep used 15 interleaved repetitions of
32 calls per kernel after warmup. Sweeps used seeds 1, 2 and 3, with the second
sweep in reverse order. The measured unit is one attention operation.

The desktop stayed active. Device-wide GPU utilization varied before each
sweep, and absolute timings varied between runs. The machine recorded
no thermal warnings and no new swapouts during the three sweeps.

## Data

| File | Contents |
| --- | --- |
| [Sweep 1](sweep-1.jsonl), [sweep 2](sweep-2.jsonl), [sweep 3](sweep-3.jsonl) | Per-case median times, executed kernels and output differences |
| [Default route checks](dispatch.jsonl) | Executed kernel and partition for each case and threshold neighbor |
| [Boundary timings](boundary-timings.jsonl) | Three P256 and established-kernel comparisons for each single-request case near the cutoff |

## Reproduce

Activate a Python 3.12 environment with the dependencies listed above.
From the repository root, put the pinned wheel first on `PYTHONPATH`
so the benchmark loads its compiled Metal kernels. The benchmark script
is identical at the measured and PR base commits.

```sh
wheel_dir=$(mktemp -d)
uv pip install --target "$wheel_dir" --no-deps \
  https://github.com/vllm-project/vllm-metal/releases/download/v0.33.0.dev20261112090227/vllm_metal-0.33.0.dev20261112090227-cp312-cp312-macosx_15_0_arm64.whl
for seed in 1 2 3; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH="$wheel_dir:$PWD" \
    python tools/benchmark/tq_gqa_decode_benchmark.py \
    --geometries 32:8:128:16 24:8:128:16 16:2:128:16 24:4:256:16 \
      24:4:256:32 16:2:256:16 16:2:256:32 \
    --contexts 2048 4096 8192 16384 --requests 1 4 \
    --dtypes float16 bfloat16 --k-quants q8_0 --v-bits 3 \
    --calls 32 --reps 15 --seed "$seed" --output "sweep-$seed.jsonl"
done
```

For the reverse sweep, reverse the lists of geometries, contexts, request
counts and dtypes for seed 2. Use fresh output files because the script appends.
To reproduce the route check, execute the same native call as the benchmark's
`established` comparison with `gqa_disabled=False`. Enable dispatch diagnostics,
evaluate the output, then read `last_paged_dispatch()` and
`last_gqa_partition_size()`. For the boundary timings, use one request
and compare only the `established` and `gqa_p256` arms. Use contexts
3,072, 5,631 and 5,632 for the 24-query-head shapes, and 3,072, 8,447 and
8,448 for the 16-query-head shapes.
