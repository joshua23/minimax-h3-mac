# reference/

Vendored upstream source, kept **only so the parity tests have something to check against**. None
of it runs at inference time — the port under `minimax_h3_mlx/` has no torch or diffusers
dependency.

## Provenance

`diffusers/` is copied verbatim from the `minimax-h3` branch of
[huggingface/diffusers](https://github.com/huggingface/diffusers/tree/minimax-h3):

| file | upstream path |
|---|---|
| `transformer_minimax_h3.py` | `src/diffusers/models/transformers/` |
| `autoencoder_kl_minimax_h3.py` | `src/diffusers/models/autoencoders/` |
| `autoencoder_kl_minimax_h3_audio.py` | `src/diffusers/models/autoencoders/` |
| `scheduling_minimax_h3.py` | `src/diffusers/schedulers/` |
| `convert_minimax_h3_to_diffusers.py` | `scripts/` |
| `modular/*.py` | `src/diffusers/modular_pipelines/minimax_h3/` |

These files are unmodified and carry their own Apache-2.0 headers, copyright The MiniMax Team and
The HuggingFace Team.

## Why they are here

The parity tests deliberately validate against the real reference rather than against a re-reading
of it. `convert_minimax_h3_to_diffusers.py` in particular is used as the *official* conversion path:
the MLX model's parameters are pushed through `reorder_interleaved_qkv` and `convert_transformer_key`
into the reference module, so the raw checkpoint's per-head-interleaved QKV and fused `[gate; value]`
SwiGLU are exercised rather than assumed.

Installing the branch itself (see `requirements.txt`) is what the tests import; the copies here make
the exact revision that was validated against legible without a checkout.
