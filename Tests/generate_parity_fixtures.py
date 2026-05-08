"""Generate parity fixtures for the Swift MLX-TIPS parity tests.

Runs the **upstream PyTorch reference** (`google-deepmind/tips`, module
`tips.pytorch`) on a fixed input and saves inputs + reference outputs as a
safetensors file that the Swift tests load via `MLX.loadArrays(url:)`.

The Swift port is parity-tested against this PyTorch fixture directly: any
shared porting bug between the Python MLX port and the Swift port would still
diverge from PyTorch and surface here. This is the canonical release gate.

Run with `uv run` from any directory; pass `PYTHONPATH=<path-to-google-deepmind/tips>`
so that `from tips.pytorch import ...` resolves (the upstream `tips` repo is not
packaged). Weights are pulled from HuggingFace by default:

    PYTHONPATH=/path/to/google-deepmind/tips \\
        uv run --with torch --with safetensors --with sentencepiece \\
              --with huggingface_hub --with pillow \\
        python Tests/generate_parity_fixtures.py --variant B --hf-repo

`--hf-repo` with no value derives `google/tipsv2-{slug}` from `--variant`. To
use weights you've already extracted as .npz files, pass `--checkpoints-dir
<dir>` instead.

Outputs are saved under both:
  - legacy top-level keys (`cls_token`, `register_tokens`, `patch_tokens`,
    `text_embeddings`) consumed by per-stage `ParityTests.swift`
  - `output.<name>` namespace consumed by end-to-end
    `ParityFixtureTests.swift`

The fixture is the same on disk; only the consuming test differs.

Output: <swift-repo>/Tests/TIPSTests/Fixtures/parity_fixtures.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

# Default checkpoint location: <swift-repo>/../../python/tips/pytorch/checkpoints
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = SCRIPT_DIR.parent.parent.parent / "python" / "tips" / "pytorch" / "checkpoints"

# Maps Swift `VisionVariant` letters to filename slugs.
# Note: HF repo names use `so400m14` while local .npz files use `so14` for SoViT.
VARIANT_NPZ_SLUG = {
    "B": "b14",
    "L": "l14",
    "So400m": "so14",
    "g": "g14",
}
VARIANT_HF_SLUG = {
    "B": "b14",
    "L": "l14",
    "So400m": "so400m14",
    "g": "g14",
}

# Text encoder per-variant config (mirrors run_text_encoder_inference.get_config).
TEXT_CONFIG = {
    "S":      {"hidden_size": 384,  "mlp_dim": 1536, "num_heads": 6,  "num_layers": 12},
    "B":      {"hidden_size": 768,  "mlp_dim": 3072, "num_heads": 12, "num_layers": 12},
    "L":      {"hidden_size": 1024, "mlp_dim": 4096, "num_heads": 16, "num_layers": 12},
    "So400m": {"hidden_size": 1152, "mlp_dim": 4304, "num_heads": 16, "num_layers": 27},
    "g":      {"hidden_size": 1536, "mlp_dim": 6144, "num_heads": 24, "num_layers": 12},
}

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
TEXTS = ["a photo of a cat", "hello world"]
MAX_LEN = 64
IMG_SIZE = 448


def make_image(image_path: str | None) -> np.ndarray:
    """Return a (1, 448, 448, 3) float32 NHWC tensor in [0, 1]."""
    if image_path is None:
        # Random pixels are sufficient for model arithmetic parity (TIPS has no
        # output-conditioned post-processing). Real images add nothing the
        # model wouldn't already see.
        return RNG.random((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, ...]  # (1, H, W, 3)


def tokenize_spm(tokenizer_path: str, texts: list[str], max_len: int = MAX_LEN):
    """Tokenize via the `sentencepiece` library directly.

    The upstream Torch reference uses `tensorflow_text.SentencepieceTokenizer`,
    but the resulting ids are identical to plain `sentencepiece` for this
    tokenizer (no BOS/EOS, lowercased input). Using `sentencepiece` avoids
    pulling TensorFlow into the fixture-generation environment.
    """
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
# PyTorch model loading & inference
# ---------------------------------------------------------------------------

def _load_checkpoint_dict(path: Path) -> dict:
    """Load weights from either a .npz or .safetensors file as a numpy dict."""
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file
        return load_file(str(path))
    return dict(np.load(str(path), allow_pickle=False))


def _split_combined_safetensors(path: Path) -> tuple[dict, dict]:
    """Split an HF-style combined safetensors checkpoint by prefix.

    HF snapshots (e.g., google/tipsv2-b14) bundle both encoders into
    `model.safetensors` with keys prefixed `vision_encoder.` / `text_encoder.`.
    Strip the prefix to get the upstream Torch state_dict layout.
    """
    raw = _load_checkpoint_dict(path)
    vision: dict = {}
    text: dict = {}
    for k, v in raw.items():
        if k.startswith("vision_encoder."):
            vision[k[len("vision_encoder."):]] = v
        elif k.startswith("text_encoder."):
            text[k[len("text_encoder."):]] = v
    return vision, text


# Parent of the upstream `tips/` clone (so `from tips.pytorch import ...` resolves
# via PEP 420 namespace packages — there is no `tips/__init__.py`).
DEFAULT_TIPS_PARENT = SCRIPT_DIR.parent.parent.parent / "python"


def run_torch(
    vision_ckpt_np: dict,
    text_ckpt_np: dict,
    tokenizer_path: Path,
    variant: str,
    image_np: np.ndarray,
    tips_path: Path,
):
    import sys
    if str(tips_path) not in sys.path:
        sys.path.insert(0, str(tips_path))

    # Stub heavy tensorflow imports — `tips.pytorch.text_encoder` imports
    # `tensorflow` and `tensorflow_text` at module scope, but only its
    # `Tokenizer` class actually uses them. We tokenize via `sentencepiece`,
    # so the stubs let the rest of the module import cleanly.
    import types
    for name in ("tensorflow", "tensorflow_text"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    import torch
    from tips.pytorch import image_encoder, text_encoder

    # ---- Vision ----
    vision_ckpt = {k: torch.from_numpy(v) for k, v in vision_ckpt_np.items()}

    model_def = {
        "S": image_encoder.vit_small,
        "B": image_encoder.vit_base,
        "L": image_encoder.vit_large,
        "So400m": image_encoder.vit_so400m,
        "g": image_encoder.vit_giant2,
    }[variant]
    ffn = "swiglu" if variant == "g" else "mlp"

    vision = model_def(
        img_size=IMG_SIZE,
        patch_size=14,
        ffn_layer=ffn,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    vision.load_state_dict(vision_ckpt)
    vision.eval()

    # NHWC (Swift fixture layout) -> NCHW (Torch layout).
    x_torch = torch.from_numpy(np.transpose(image_np, (0, 3, 1, 2)).copy())
    with torch.no_grad():
        feats = vision.forward_features(x_torch)
    cls     = feats["x_norm_1st_clstoken"].detach().numpy()
    regs    = feats["x_norm_2nd_clstoken"].detach().numpy()
    patches = feats["x_norm_patchtokens"].detach().numpy()

    # ---- Text ----
    text_ckpt = {k: torch.from_numpy(v) for k, v in text_ckpt_np.items()}
    for stale in ("temperature", "temperature_contrastive"):
        text_ckpt.pop(stale, None)

    text_model = text_encoder.TextEncoder(TEXT_CONFIG[variant], vocab_size=32_000)
    text_model.load_state_dict(text_ckpt)
    text_model.eval()

    ids_np, pads_np = tokenize_spm(str(tokenizer_path), TEXTS)
    with torch.no_grad():
        txt = text_model(torch.from_numpy(ids_np), torch.from_numpy(pads_np)).detach().numpy()

    return cls, regs, patches, ids_np, pads_np, txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_inputs(args) -> tuple[dict, dict, Path, str]:
    """Return (vision_ckpt_np, text_ckpt_np, tokenizer_path, source_label)."""
    snap_path: Path | None = None
    if args.hf_repo:
        from huggingface_hub import snapshot_download
        repo_id = args.hf_repo if "/" in args.hf_repo else f"google/tipsv2-{VARIANT_HF_SLUG[args.variant]}"
        snap_path = Path(snapshot_download(repo_id=repo_id))
    elif args.snapshot:
        snap_path = Path(args.snapshot).expanduser()

    if snap_path is not None:
        combined = snap_path / "model.safetensors"
        tokenizer = Path(args.tokenizer) if args.tokenizer else snap_path / "tokenizer.model"
        if not combined.exists():
            raise FileNotFoundError(f"HF snapshot is missing model.safetensors: {combined}")
        if not tokenizer.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer}")
        vision, text = _split_combined_safetensors(combined)
        return vision, text, tokenizer, str(snap_path)

    ckpt_dir = Path(args.checkpoints_dir).expanduser()
    slug = VARIANT_NPZ_SLUG[args.variant]
    vision_path = Path(args.vision_npz) if args.vision_npz else ckpt_dir / f"tips_v2_oss_{slug}_vision.npz"
    text_path   = Path(args.text_npz)   if args.text_npz   else ckpt_dir / f"tips_v2_oss_{slug}_text.npz"
    tokenizer   = Path(args.tokenizer)  if args.tokenizer  else ckpt_dir / "tokenizer.model"

    for p in (vision_path, text_path, tokenizer):
        if not p.exists():
            raise FileNotFoundError(f"Required checkpoint file not found: {p}")

    return (
        _load_checkpoint_dict(vision_path),
        _load_checkpoint_dict(text_path),
        tokenizer,
        f"{vision_path} + {text_path}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=list(VARIANT_HF_SLUG))
    ap.add_argument("--hf-repo", default=None, nargs="?", const="auto",
                    help="HuggingFace repo id (e.g., google/tipsv2-b14). "
                         "Pass `auto` (or just `--hf-repo` with no value) to derive from --variant. "
                         "Uses huggingface_hub.snapshot_download to fetch.")
    ap.add_argument("--snapshot", default=None,
                    help="HF snapshot directory containing model.safetensors + tokenizer.model "
                         "(e.g., ~/.cache/huggingface/hub/models--google--tipsv2-b14/snapshots/<sha>). "
                         "Skipped when --hf-repo is given.")
    ap.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS),
                    help="Directory containing tips_v2_oss_*_vision.{npz,safetensors} + _text.* + tokenizer.model. "
                         "Used when --snapshot is not given.")
    ap.add_argument("--vision-npz", default=None, help="Override vision checkpoint path")
    ap.add_argument("--text-npz", default=None, help="Override text checkpoint path")
    ap.add_argument("--tokenizer", default=None, help="Override tokenizer.model path")
    ap.add_argument("--tips-path", default=str(DEFAULT_TIPS_PARENT),
                    help="Directory whose subdir `tips/pytorch/` is the upstream "
                         "google-deepmind/tips clone. Added to sys.path so "
                         "`from tips.pytorch import ...` resolves "
                         "(PEP 420 namespace package — no tips/__init__.py).")
    ap.add_argument("--image", default=None,
                    help="Optional path to a real input image. "
                         "If omitted, deterministic random pixels are used "
                         "(sufficient for model arithmetic parity).")
    args = ap.parse_args()

    vision_ckpt, text_ckpt, tokenizer, source_label = _resolve_inputs(args)

    print(f"Variant:    {args.variant}")
    print(f"Source:     {source_label}")
    print(f"Tokenizer:  {tokenizer}")
    print(f"Image:      {args.image or 'random (seed=42)'}")

    image_np = make_image(args.image)
    print("Running PyTorch reference…")
    cls, regs, patches, ids_np, pads_np, txt = run_torch(
        vision_ckpt, text_ckpt, tokenizer, args.variant, image_np,
        Path(args.tips_path).expanduser(),
    )

    out_dir = SCRIPT_DIR.parent / "Tests" / "TIPSTests" / "Fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity_fixtures.safetensors"

    from safetensors.numpy import save_file
    save_file(
        {
            # inputs
            "image_input":           image_np,
            "text_ids":              ids_np.astype(np.float32),   # safetensors has no int32; cast for storage
            "text_paddings":         pads_np.astype(np.float32),
            # vision outputs (legacy keys for per-stage ParityTests.swift)
            "cls_token":             cls,
            "register_tokens":       regs,
            "patch_tokens":          patches,
            # text output
            "text_embeddings":       txt,
            # end-to-end pipeline outputs (consumed by ParityFixtureTests.swift)
            "output.cls_token":       cls,
            "output.register_tokens": regs,
            "output.patch_tokens":    patches,
            "output.text_embeddings": txt,
        },
        str(out_path),
    )
    print(f"Saved fixtures → {out_path}")
    print(f"  image_input:      {image_np.shape}")
    print(f"  cls_token:        {cls.shape}")
    print(f"  register_tokens:  {regs.shape}")
    print(f"  patch_tokens:     {patches.shape}")
    print(f"  text_embeddings:  {txt.shape}")


if __name__ == "__main__":
    main()
