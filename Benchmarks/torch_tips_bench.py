"""PyTorch reference benchmark for the TIPS image encoder.

Mirrors the protocol of `Tools/tips-bench/main.swift` so the two are directly
comparable: same iteration count, same image, same model variant, same dtype.

Usage:
    PYTHONPATH=/path/to/google-deepmind/tips/parent \\
        uv run --with torch --with safetensors --with huggingface_hub \\
              --with pillow --with numpy \\
        python Benchmarks/torch_tips_bench.py \\
        --variant B --hf-repo --input <image.jpg>

Modes:
    inference-only (default): the input image is preloaded once; the timed
        loop runs only the model forward pass. Synchronizes via
        `torch.mps.synchronize()` (or CUDA equivalent) before/after each iter.
    --include-load:           the timed loop also reloads the input from disk
        and uploads it to the device.

Output: JSON dict on stdout (and a human-readable summary on stderr) so
results can be machine-aggregated into a table.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Same conventions as the fixture generator; keep them aligned.
# ---------------------------------------------------------------------------

VARIANT_HF_SLUG = {
    "B": "b14", "L": "l14", "So400m": "so400m14", "g": "g14",
}

IMG_SIZE = 448
PATCH_SIZE = 14


def stub_tensorflow():
    """tips.pytorch.text_encoder imports tensorflow at module scope; we don't
    need the Tokenizer class, so stub the imports out."""
    for name in ("tensorflow", "tensorflow_text"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def split_combined_safetensors(path: Path) -> tuple[dict, dict]:
    from safetensors.numpy import load_file
    raw = load_file(str(path))
    vision: dict = {}
    text: dict = {}
    for k, v in raw.items():
        if k.startswith("vision_encoder."):
            vision[k[len("vision_encoder."):]] = v
        elif k.startswith("text_encoder."):
            text[k[len("text_encoder."):]] = v
    return vision, text


def resolve_snapshot(args) -> Path:
    if args.snapshot:
        return Path(args.snapshot).expanduser()
    if args.hf_repo:
        from huggingface_hub import snapshot_download
        repo_id = args.hf_repo if "/" in args.hf_repo else f"google/tipsv2-{VARIANT_HF_SLUG[args.variant]}"
        return Path(snapshot_download(repo_id=repo_id))
    raise SystemExit("Pass --hf-repo or --snapshot.")


def load_image_path(path: Path) -> np.ndarray:
    """Load image, resize to (448, 448), return NCHW float32 in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    return np.transpose(arr, (2, 0, 1))[None, ...]    # 1xCxHxW


def make_model(variant: str, vision_ckpt: dict, device, dtype):
    import torch
    from tips.pytorch import image_encoder

    model_def = {
        "B": image_encoder.vit_base,
        "L": image_encoder.vit_large,
        "So400m": image_encoder.vit_so400m,
        "g": image_encoder.vit_giant2,
    }[variant]
    ffn = "swiglu" if variant == "g" else "mlp"

    model = model_def(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        ffn_layer=ffn, block_chunks=0, init_values=1.0,
        interpolate_antialias=True, interpolate_offset=0.0,
    )
    state = {k: torch.from_numpy(v) for k, v in vision_ckpt.items()}
    model.load_state_dict(state)
    model.eval().to(device=device, dtype=dtype)
    return model


def synchronize(device):
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=list(VARIANT_HF_SLUG))
    ap.add_argument("--hf-repo", default=None, nargs="?", const="auto")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--input", required=True, help="Path to input image.")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--include-load", action="store_true",
                    help="End-to-end mode: also time disk read + device upload.")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--tips-path", default=str(Path(__file__).resolve().parent.parent.parent.parent / "python"),
                    help="Parent of the upstream google-deepmind/tips clone (added to sys.path).")
    args = ap.parse_args()

    sys.path.insert(0, args.tips_path)
    stub_tensorflow()

    import torch
    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # ---- Cold model load ----
    snap = resolve_snapshot(args)
    print(f"snapshot: {snap}", file=sys.stderr)
    print(f"device:   {device}", file=sys.stderr)

    t0 = time.perf_counter()
    vision_ckpt, _ = split_combined_safetensors(snap / "model.safetensors")
    model = make_model(args.variant, vision_ckpt, device, dtype)
    synchronize(device)
    load_s = time.perf_counter() - t0

    image_path = Path(args.input)

    # ---- Preload (when not --include-load) ----
    if not args.include_load:
        x_np = load_image_path(image_path)
        x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
        synchronize(device)

    def run_once() -> float:
        # Drain any tail from the previous iteration before starting the timer.
        synchronize(device)
        t = time.perf_counter()
        if args.include_load:
            x_local_np = load_image_path(image_path)
            x_local = torch.from_numpy(x_local_np).to(device=device, dtype=dtype)
        else:
            x_local = x  # noqa: F821
        with torch.no_grad():
            _ = model.forward_features(x_local)
        synchronize(device)
        return time.perf_counter() - t

    # ---- Warmup ----
    for _ in range(args.warmup):
        run_once()

    # ---- Timed loop ----
    times = [run_once() for _ in range(args.iters)]

    result = {
        "backend":   f"torch-{device.type}",
        "variant":   args.variant,
        "mode":      "end-to-end" if args.include_load else "inference-only",
        "dtype":     args.dtype,
        "iters":     args.iters,
        "warmup":    args.warmup,
        "load_s":    load_s,
        "times":     times,
        "mean_s":    statistics.fmean(times),
        "median_s":  statistics.median(times),
        "min_s":     min(times),
        "max_s":     max(times),
    }

    print(json.dumps(result))
    print(
        f"\n[{result['backend']}/{result['mode']}/{result['dtype']}]  "
        f"median={result['median_s']*1000:.1f} ms  "
        f"mean={result['mean_s']*1000:.1f} ms  "
        f"min={result['min_s']*1000:.1f} ms  "
        f"max={result['max_s']*1000:.1f} ms  "
        f"load={result['load_s']:.2f} s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
