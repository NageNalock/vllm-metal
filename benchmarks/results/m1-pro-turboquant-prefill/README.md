# M1 Pro TurboQuant prefill measurements

These runs compare the existing compressed and opt-in tiled attention paths on one M1 Pro. The code fix reads strided page and offset coordinates correctly when materializing KV. It addresses a numerical error in the single-page crossover case. The larger-prompt speedups measure the choice of attention path.

The upstream base is `3a37a22abc08c1bbdec2f64433e66ef7534c9b69`. The fix is `97bc37bec11e75f8c50b1d831a08e7a8f61a859a`. Both used the upstream base's released native artifacts and MLX 0.32.4. MLX compiled the materializer from the respective Python source at runtime. The full-model runs used the same local Qwen3-4B-4bit checkpoint at revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. [Environment and workload settings](environment.json)

| Files | Contents |
|---|---|
| [Single-layer records](single-layer.jsonl) | All 31 measured times per arm in each of 30 cases, maximum numerical error, actual lane selection and incremental MLX allocation. |
| [Base offline measurements](offline-base.json) and [fixed offline measurements](offline-fixed.json) | Warming and measured requests, input hashes, generated token IDs, TTFT, MLX allocation and planner counts. Model loading precedes the measured requests. |
| [Compressed HTTP run](http-compressed.json) and [tiled HTTP run](http-tiled.json) | Upstream `vllm bench serve --save-detailed` results from two fresh servers. The client used the same random dataset and seed in each run. |
| [Dense reference](oracle.json) and [reference script](dense_reference.py) | A second check that decodes the paged K/V with MLX and computes causal attention in NumPy FP32. `lane_err` is the maximum absolute error of forced materialization against that reference. `delta` compares forced and compressed outputs. |
| [Summary](summary.json) | Median latency, MLX allocation above each request's starting level, output agreement and the single-layer ratios. |

The single-page regression and both strided-offset checks failed on the base wheel. The fixed checkout passed all 463 tests across the six TurboQuant, routing and benchmark test files used for review. Ruff, its formatter, ShellCheck and mypy passed.

## Reproduce

Use the checkout corresponding to the run and its native artifacts. Run the commands from the repository root with the virtual environment active. The machine was on AC power with Low Power Mode off. GPU utilization before the run ranged from 16% to 24% over three samples on an interactive desktop.

```bash
export PYTHONPATH=. VLLM_METAL_BACKEND=mlx MLX_ENABLE_TF32=0 PYTORCH_ENABLE_MPS_FALLBACK=0
export VLLM_METAL_TQ_PREFILL=1
python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --tiled \
  --head-pairs 32:8 --query-tokens 16 32 64 128 256 \
  --context-tokens 0 8192 32768 --reps 31 --warmup 5
python tools/benchmark/tq_e2e_arm.py --model mlx-community/Qwen3-4B-4bit \
  --arm paired --prompt-tokens 512 2048 --batch-tokens 2048 --max-tokens 8 \
  --gpu-memory-utilization 0.4 --warmup 1 --reps 3 --output offline.json
```

For the original failure, use `--query-tokens 16 --context-tokens 0 --warmup 0 --reps 1` in the first command. For a single 16-token page, the page index vector has byte stride 0 and the offset vector has byte stride 4. Making just the page index vector contiguous on the base reduced the FP16 difference between the two attention paths from 3.071045 to 0.001953. With the fix the original view gives the same 0.001953 difference. The ordinary short-prompt policy selected the compressed path.

For the HTTP test, launch one server at a time with `mode=0` for compressed and then `mode=1` for tiled. Run the same client command against each ready server.

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_METAL_TQ_PREFILL=$mode \
  vllm serve mlx-community/Qwen3-4B-4bit --host 127.0.0.1 --served-model-name tq-prefill \
  --dtype bfloat16 --max-model-len 2056 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.4 --no-enable-prefix-caching --generation-config vllm \
  --additional-config '{"turboquant":true,"k_quant":"q8_0","v_quant":"q3_0"}'
vllm bench serve --model mlx-community/Qwen3-4B-4bit --served-model-name tq-prefill \
  --backend openai --endpoint /v1/completions --dataset-name random --random-input-len 2048 \
  --random-output-len 8 --random-range-ratio 0 --num-prompts 3 --num-warmups 1 \
  --max-concurrency 1 --ignore-eos --temperature 0 --seed 853 --save-result --save-detailed
```
