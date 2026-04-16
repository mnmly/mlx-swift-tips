"""Generate parity fixtures for the Swift MLX-TIPS parity tests.

Loads the Python mlx_tips model, runs fixed inputs through vision and text
encoders, and saves inputs + reference outputs as a .npz file that the Swift
tests can load via swift-npy.

Run from ../../python/mlx-tips (the Python package root):
    uv run python ../../swift/mlx-swift-tipsv2/tests/generate_parity_fixtures.py
    uv run python ../../swift/mlx-swift-tipsv2/tests/generate_parity_fixtures.py --variant B --snapshot /path/to/snapshot

Output: <swift-repo>/Tests/TIPSTests/Fixtures/parity_fixtures.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Snapshot discovery (mirrors test_parity.py)
# ---------------------------------------------------------------------------

HF_CACHE_ROOT = Path(os.environ.get(
    "HF_CACHE_ROOT",
    Path("~/.cache/huggingface/hub")
))
REPO_DIRS = {
    "B": "models--google--tipsv2-b14",
}


def find_snapshot(variant: str) -> Path:
    repo = HF_CACHE_ROOT / REPO_DIRS[variant] / "snapshots"
    snapshots = sorted(repo.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {repo}")
    return max(snapshots, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Fixed inputs (seeded for reproducibility)
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
IMAGE_NP = RNG.random((1, 448, 448, 3), dtype=np.float32)
TEXTS = ["a photo of a cat", "hello world"]
MAX_LEN = 64


def tokenize_spm(tokenizer_path: str, texts: list[str], max_len: int = MAX_LEN):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    tokens = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = sp.encode(t.lower(), add_bos=False, add_eos=False)
        n = min(len(ids), max_len)
        tokens[i, :n] = ids[:n]
    paddings = (tokens == 0).astype(np.int32)
    return tokens, paddings


# ---------------------------------------------------------------------------
# MLX model loading & inference
# ---------------------------------------------------------------------------

def run_mlx(snap: Path, variant: str):
    import mlx.core as mx
    # mlx_tips is importable when run via `uv run` from the python/mlx-tips directory.
    from mlx_tips import (
        build_vision_transformer,
        build_text_encoder,
        _build_vision_weights,
        _build_text_weights,
    )

    raw = mx.load(str(snap / "model.safetensors"))

    vision = build_vision_transformer(variant, img_size=448)
    vision.load_weights(_build_vision_weights(raw))
    mx.eval(vision.parameters())

    text = build_text_encoder(variant)
    text.load_weights(_build_text_weights(raw))
    mx.eval(text.parameters())

    # Vision
    x = mx.array(IMAGE_NP)
    vis_out = vision(x)
    mx.eval(vis_out.cls_token, vis_out.register_tokens, vis_out.patch_tokens)
    cls     = np.asarray(vis_out.cls_token)
    regs    = np.asarray(vis_out.register_tokens)
    patches = np.asarray(vis_out.patch_tokens)

    # Text
    ids_np, pads_np = tokenize_spm(str(snap / "tokenizer.model"), TEXTS)
    ids  = mx.array(ids_np.astype(np.int32))
    pads = mx.array(pads_np.astype(np.int32))
    txt_emb = text(ids, pads)
    mx.eval(txt_emb)
    txt = np.asarray(txt_emb)

    return cls, regs, patches, ids_np, pads_np, txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=["B"])
    ap.add_argument("--snapshot", default=None,
                    help="Explicit path to snapshot dir (overrides auto-discovery)")
    args = ap.parse_args()

    snap = Path(args.snapshot) if args.snapshot else find_snapshot(args.variant)
    print(f"Snapshot: {snap}")

    print("Running MLX model…")
    cls, regs, patches, ids_np, pads_np, txt = run_mlx(snap, args.variant)

    out_dir = Path(__file__).parent.parent / "Tests" / "TIPSTests" / "Fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity_fixtures.safetensors"

    from safetensors.numpy import save_file
    save_file(
        {
            # inputs
            "image_input":     IMAGE_NP,
            "text_ids":        ids_np.astype(np.float32),   # safetensors has no int32; cast for storage
            "text_paddings":   pads_np.astype(np.float32),
            # vision outputs
            "cls_token":       cls,
            "register_tokens": regs,
            "patch_tokens":    patches,
            # text output
            "text_embeddings": txt,
        },
        str(out_path),
    )
    print(f"Saved fixtures → {out_path}")
    print(f"  image_input:      {IMAGE_NP.shape}")
    print(f"  cls_token:        {cls.shape}")
    print(f"  register_tokens:  {regs.shape}")
    print(f"  patch_tokens:     {patches.shape}")
    print(f"  text_embeddings:  {txt.shape}")


if __name__ == "__main__":
    main()
