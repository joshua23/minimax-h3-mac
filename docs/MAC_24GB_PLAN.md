# MiniMax-H3 on a 24 GB Apple Silicon Mac

## Target machine

- Apple M4 Pro, 20-core GPU, 24 GB unified memory
- Metal 3, macOS 15.1
- MLX maximum recommended working set: 17.18 GB
- Practical project ceiling: 15-16 GB per phase, with the MLX free-buffer cache disabled

## What is and is not transferable

Enze Xie's relevant work is SANA, SANA 1.5, SANA-Sprint, and SANA-Video. Its major gains come from an architecture trained with linear/block-causal attention and consistency distillation. Those weights and kernels are not a drop-in replacement for H3's trained dense self-attention.

The directly reusable NVIDIA reference is LightX2V's MiniMax-H3 runner:

- independent text-encoder, transformer, and VAE residency;
- block-level double-buffered weight prefetch;
- FP8/INT8 transformer weights;
- tensor/sequence parallelism;
- SageAttention and block-sparse attention.

On Apple Silicon, tensor/sequence parallelism and CUDA/Triton/Sage kernels do not transfer. Component staging, block streaming, weight quantization, residual caching, and few-step distillation do.

## Measured M4 Pro baseline

The upstream MLX smoke suite passes without model weights. A real-size H3 transformer block was benchmarked with hidden size 5376, 56x128 heads, FFN 14336, and bfloat16 compute:

| Packed rows | One block | 50-block step | Four steps | Peak MLX memory |
|---:|---:|---:|---:|---:|
| 512 | 0.350 s | 17.5 s | 1.2 min | 3.9 GB |
| 2,048 | 0.268 s | 13.4 s | 0.9 min | 3.9 GB |
| 8,192 | 1.372 s | 68.6 s | 4.6 min | 3.9 GB |
| 16,384 | 3.182 s | 2.65 min | 10.6 min | 4.8 GB |
| 37,966 | 12.113 s | 10.1 min | 40.4 min | 7.7 GB |

37,966 rows is the released five-second 1344x768 geometry. Around 8K-10K rows is a useful estimate for a five-second 672x384 proof run.

The result is decisive: compute is slow but viable. Eager weight residency is the deployment blocker.

## Required memory changes

The released pipeline components are approximately:

- DiT: 66.3 GB mixed BF16/FP32; published MLX 4-bit transformer is 25.3 GB on disk and 11.5 GB resident after AdaLN drop.
- Truncated Qwen3-VL conditioner: 50.3 GB resident in BF16.
- Video VAE: 10.4 GB.
- Audio VAE: 0.6 GB.

They must never coexist in memory on this Mac.

### Phase 1: staged components

1. Load a 4-bit truncated text encoder only.
2. Produce prompt embeddings and text modality tags.
3. Materialize the small outputs, destroy the encoder, synchronize, and call `mx.clear_cache()`.
4. Load the video VAE only when keyframes need encoding; unload it immediately afterward.
5. Run the DiT phase.
6. Destroy the DiT, then load/decode video and audio VAEs one at a time.

The current upstream MLX pipeline eagerly loads all four components and must be refactored.

### Phase 2: streamed AdaLN precompute

Published quantized builds still need the 13B AdaLN projections briefly before dropping them. Loading the whole 25.3 GB quantized transformer exceeds the machine budget.

For a fixed sigma schedule, stream one block's 8-bit AdaLN projection from safetensors, compute its modulation rows, retain only the resulting table, and discard the projection before loading the next block. The final table is about 145 MB for nine schedule points.

### Phase 3: streamed transformer blocks

Create one reusable quantized `TransformerBlock` and a block index over the safetensors shards. For each of 50 blocks:

1. lazily load only that block's 4-bit weights;
2. evaluate the block against the current hidden state;
3. materialize the hidden state;
4. replace the reusable block weights and reclaim the old buffers.

MLX safetensor arrays are lazy file-backed loads, and MLX supplies memory/cache/wired limits. Apple unified memory removes a separate host-to-device copy; the useful overlap is SSD read of block N+1 with Metal compute of block N.

This reduces transformer weight residency from 11.5 GB to roughly one quantized block plus static pre/post weights. It is required for 768p because measured activations already peak at 7.7 GB.

## Step-count and block-count acceleration

### H3-Turbo

`lightx2v/Minimax-h3-Turbo` publishes a preview four-step LoRA. It targets the q/k/v/out and FFN projections. A streaming converter can fuse each LoRA delta into one source tensor and quantize that tensor immediately, without ever materializing the 66 GB base model.

The current preview reports weaker fine detail, but it changes a practical 768p estimate from multiple hours to about 40 minutes of dense DiT compute on this M4 Pro before other optimizations.

### MLX block residual cache

Implemented in this repository. The cache:

- always executes the first and last denoising steps fully;
- recomputes a configurable warm prefix;
- reuses the prior full step's trailing-block residual when adjacent sigmas are close;
- forces periodic full refreshes;
- is disabled by default and bit-exact when disabled.

Tiny-config tests prove exact fail-off behavior and actual block skipping. The NVIDIA/ComfyUI reference reports about 45% block-compute savings on a 30-step workflow. Four-step Turbo has fewer safe cache opportunities, so it should be benchmarked rather than assumed additive.

## Attention work

MLX already uses its optimized Steel flash-style SDPA and does not materialize the score matrix. Rewriting the same dense attention as a custom Metal kernel is unlikely to alter the fundamental cost.

The high-value experimental target is structured sparsity:

1. collect teacher dense attention block statistics on low-resolution trajectories;
2. test static 3D sliding-tile, radial, and modality-aware patterns;
3. preserve global text rows and selected audio/video sink blocks;
4. use exact dense attention for early/late steps and sparse attention in the middle;
5. compare per-step velocity relative L2/cosine under teacher forcing before rendering;
6. only then implement a block-LUT Metal kernel with `mx.fast.metal_kernel`.

Candidate methods to study are Sliding Tile Attention, Sparse VideoGen, AdaSpa, SpargeAttention, Radial Attention, GRAT, and LightX2V's dynamic Sparse SageAttention. Methods requiring retraining are later work; training-free masks and routing come first.

## Delivery ladder

1. **Done:** MLX smoke on M4 Pro and real-size block benchmarks.
2. **Done:** configurable MLX block-tail residual cache with exact disabled path.
3. **Next:** 4-bit truncated Qwen3-VL converter/loader and staged component lifetime.
4. **Next:** streamed AdaLN modulation table.
5. **Next:** one-block-at-a-time 4-bit DiT runner.
6. **Then:** tensor-by-tensor H3-Turbo LoRA fusion into the 4-bit stream checkpoint.
7. **Proof target:** 5 seconds, 672x384, four transformer evaluations.
8. **Deployment target:** 5 seconds, 1344x768, four evaluations, below 16 GB peak.
9. **Research target:** training-free block-sparse Metal attention with teacher-forced quality gates.

## Expected runtime

Based on measured DiT blocks, before VAE and I/O overhead:

- 384p-class proof: about 5 minutes for four dense evaluations;
- 768p five-second clip: about 40 minutes for four dense evaluations;
- cache may reduce this further, but quality and hit rate must be measured;
- 2K remains out of scope because H3-Regenerate-2K is not open-sourced and local dense compute is excessive.

## License boundaries

- MiniMax weights use the MiniMax H3 Community License, not an OSI open-source license.
- The MLX port code is Apache-2.0; derivative weights retain the upstream weight-license obligations.
- H3-Turbo code is Apache-2.0, but its LoRA is derived from H3 and does not remove upstream obligations.
- The clean-room MLX cache implementation does not copy GPL ComfyUI code.

Downloading the model weights constitutes acceptance of the upstream license and should be an explicit operator action.
