"""Create the Hugging Face collection that groups the MiniMax-H3 MLX builds.

Each item carries a note saying what it is *for*, since the widths differ in ways a size column does
not convey — and one width is deliberately absent, which is worth stating where people will look.

    ./.venv/bin/python scripts/make_collection.py --yes
"""

from __future__ import annotations

import argparse

NAMESPACE = "pipenetwork"
TITLE = "MiniMax-H3 MLX"

# The collection description is capped at 150 characters, so the substance lives in the
# model cards and in the per-item notes below.
DESCRIPTION = (
    "Apple Silicon (MLX) builds of MiniMax-H3, the 33B joint video+audio diffusion "
    "transformer. Code: github.com/PipeNetwork/minimax-h3-mlx"
)

# Largest first, so the list reads as a quality ladder.
ITEMS = [
    ("pipenetwork/MiniMax-H3-MLX-f32",
     "132.6 GB. Upcast float32. No more information than bf16 - for fine-tuning, not for generating."),
    ("pipenetwork/MiniMax-H3-MLX-bf16",
     "66.3 GB. Unquantized, the release's native mixed bf16/f32 precision. The quality reference."),
    ("pipenetwork/MiniMax-H3-MLX-8bit",
     "35.3 GB download, 21.5 GB resident. 27.6 dB PSNR vs bfloat16 - near-indistinguishable."),
    ("pipenetwork/MiniMax-H3-MLX-6bit",
     "30.3 GB download, 16.5 GB resident. Bracketed by two widths verified by generation."),
    ("pipenetwork/MiniMax-H3-MLX-4bit",
     "25.3 GB download, 11.5 GB resident. 22.0 dB PSNR - artifacts visible, subject intact."),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--yes", action="store_true", help="actually create it")
    args = parser.parse_args()

    print(f"collection: {NAMESPACE}/{TITLE}{' (private)' if args.private else ''}")
    print(f"\n{DESCRIPTION}\n")
    for item, note in ITEMS:
        print(f"  {item}\n     {note}")

    if not args.yes:
        print("\ndry run — pass --yes to create")
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi()
    collection = api.create_collection(
        title=TITLE,
        namespace=NAMESPACE,
        description=DESCRIPTION,
        private=args.private,
        exists_ok=True,
    )
    added, skipped = 0, []
    for item, note in ITEMS:
        # A repo may not be published yet — add what exists rather than failing the whole refresh,
        # so a partial set still produces a usable collection.
        try:
            api.model_info(item)
        except (RepositoryNotFoundError, HfHubHTTPError):
            skipped.append(item)
            continue
        api.add_collection_item(
            collection.slug, item_id=item, item_type="model", note=note, exists_ok=True
        )
        added += 1

    print(f"\nhttps://huggingface.co/collections/{collection.slug}")
    print(f"added {added} item(s)")
    if skipped:
        print(f"skipped {len(skipped)} not yet published: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
