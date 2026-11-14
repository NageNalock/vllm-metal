# MoE Expert Offloading

Mixture-of-experts checkpoints are mostly expert weights, but each token runs
only `top_k` of them. With expert offloading, the experts stay in the
checkpoint on disk and each MoE layer reads the experts its tokens route to,
through a cache in unified memory. This serves MoE models that do not fit in
memory when fully loaded.

The layer stacks the experts a step touches and uses mlx-lm's expert
projections with the original expert weights and routing.

## Quick Start

```bash
vllm serve mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --additional-config '{"moe_expert_offload": {"cache_gb": 1.5}}'
```

| Key | Description |
|-----|-------------|
| `moe_expert_offload.cache_gb` | Memory for cached experts, in GiB. Required to enable offloading. |
| `moe_expert_offload.max_experts_per_pass` | Optional positive integer. Bounds how many experts are loaded and stacked for each group within a layer. |

The KV cache budget reserves `cache_gb` plus temporary expert weights before
sizing. By default the reservation covers a full layer, since a long prefill
can touch nearly all of its experts. With `max_experts_per_pass`, the temporary
reservation covers one group's loaded weights and its packed stack. The layer
processes larger routed sets one group at a time, which can increase the
time to the first token. Steps that stay within the limit execute in one pass.
Grouped passes merge adjacent reads in the original checkpoint to reduce
the number of requests to the SSD.

For Qwen3-Coder-Next 4-bit, a full layer's expert stack is 864 MiB. A limit
of 128 experts reduces that stack to 216 MiB, with a total temporary
reservation of 432 MiB that also covers the individual expert arrays.

## Choosing the cache size

The cache holds the most recently used experts across all layers. With a
single request, each decode step reads `top_k` experts per layer; a cache that
holds a few steps' worth keeps most reads warm. Prefill and batched decode
route to many more experts, so each step reads more from disk. Offloading
targets one to a few concurrent requests on Macs that cannot hold the model.

Expect lower decode throughput than a resident model, because every MoE layer
waits for its routing before it reads experts, and misses read from the SSD.

## Supported checkpoints

- mlx-lm text models whose experts are mlx-lm `SwitchGLU` layers, in two
  safetensors layouts: experts stacked under `<block>.switch_mlp` (mlx-community
  conversions such as Qwen3-30B-A3B) and one tensor per expert under
  `<block>.experts.<j>` (OLMoE and Qwen1.5-MoE conversions).
- Shared experts, attention and embeddings stay resident.

Not supported: the MPS backend, GGUF, AWQ and multimodal checkpoints,
pipeline parallelism, LoRA, speculative decoding, and checkpoints that name
expert projections differently (Mixtral's `w1`/`w2`/`w3`).
