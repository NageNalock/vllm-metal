"""Compare three attention paths to a dense FP32 causal reference on the same KV."""

import json

import mlx.core as mx
import numpy as np

from tools.benchmark.tq_lane_verify import run
from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.caches.turboquant import turbo_quant_decode
from vllm_metal.attention.impls import sdpa
from vllm_metal.metal import get_ops


def as_float32(value):
    return np.asarray(value.astype(mx.float32))


def max_error(output, reference):
    return float(np.max(np.abs(output - reference)))


def compare(query_length, context_length):
    case = build_case(
        qlens=(query_length,),
        context_lens=(context_length,),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        dtype=mx.float16,
    )
    ops = get_ops()
    compressed = as_float32(run(case, True))
    compressed_family = ops.last_paged_dispatch()
    production = as_float32(run(case, False))
    production_family = ops.last_paged_dispatch()
    tiled = as_float32(run(case, False, force_materialization=True))
    tiled_family = ops.last_paged_dispatch()

    cache = case.cache
    block_size = cache.block_size
    pages = mx.array(
        [case.ctx.block_tables[0][i // block_size] for i in range(context_length)]
    )
    offsets = mx.array([i % block_size for i in range(context_length)])

    def gather(fields):
        return fields[0][pages, offsets]

    keys, values = turbo_quant_decode(
        (
            gather(cache.key_caches),
            gather(cache.key_scale_caches),
            gather(cache.key_zero_caches),
        ),
        (gather(cache.value_caches), gather(cache.value_scale_caches)),
        output_dtype=mx.float16,
        key_quant_type=cache.k_quant,
        value_bits=cache.v_bits,
    )
    queries = sdpa.prepare_sdpa_qkv(case.inner, case.x, case.ctx, 32, 8)[0]
    # Expand the eight KV heads to the corresponding 32 query heads.
    keys = np.repeat(as_float32(keys).transpose(1, 0, 2), 4, axis=0)
    values = np.repeat(as_float32(values).transpose(1, 0, 2), 4, axis=0)
    scores = (as_float32(queries)[0] @ keys.swapaxes(-1, -2)) * case.inner.scale
    positions = np.arange(context_length - query_length, context_length)
    scores = np.where(
        np.arange(context_length)[None, :] <= positions[:, None], scores, -np.inf
    )
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    dense = (probabilities @ values).transpose(1, 0, 2).reshape(1, query_length, -1)
    return {
        "q": query_length,
        "s": context_length,
        "old_family": compressed_family,
        "prod_family": production_family,
        "lane_family": tiled_family,
        "old_err": max_error(compressed, dense),
        "prod_err": max_error(production, dense),
        "lane_err": max_error(tiled, dense),
        "delta": max_error(compressed, tiled),
        "bad_last_rows": [
            max_error(output[:, -1], dense[:, -1]) for output in (compressed, tiled)
        ],
    }


if __name__ == "__main__":
    get_ops().set_nax_enabled(False)
    get_ops()._set_paged_dispatch_diagnostics(True)
    for query, context in (
        (16, 16),
        (16, 17),
        (16, 31),
        (16, 32),
        (16, 64),
        (16, 257),
        (32, 32),
        (64, 64),
        (128, 128),
        (15, 15),
        (17, 17),
        (31, 31),
        (33, 33),
        (8, 8),
        (2, 2),
        (16, 8192),
    ):
        print(json.dumps(compare(query, context)), flush=True)
