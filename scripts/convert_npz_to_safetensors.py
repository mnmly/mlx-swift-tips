#!/usr/bin/env python3
"""Convert TIPS `.npz` checkpoints to `.safetensors` for the Swift loader.

MLX Swift's ``loadArrays`` only accepts ``.safetensors``, so the split
vision/text ``.npz`` files from Google Storage must be converted once.

Usage:
    python scripts/convert_npz_to_safetensors.py checkpoints/*.npz

Each ``foo.npz`` becomes ``foo.safetensors`` alongside it. Scalar entries
(e.g. the contrastive ``temperature``) and any 0-d arrays are dropped,
matching the Python MLX loader's behavior.

Requires: ``mlx``, ``numpy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert(path: Path) -> Path:
    if path.suffix != ".npz":
        raise ValueError(f"not a .npz file: {path}")
    out = path.with_suffix(".safetensors")
    print(f"converting {path.name} -> {out.name}")

    with np.load(path, allow_pickle=False) as z:
        arrays: dict[str, mx.array] = {}
        dropped: list[str] = []
        for k in z.files:
            v = z[k]
            if v.ndim == 0:
                dropped.append(k)
                continue
            arrays[k] = mx.array(v)

    if dropped:
        print(f"  dropped scalar keys: {dropped}")
    mx.save_safetensors(str(out), arrays)
    print(f"  wrote {len(arrays)} tensors, {out.stat().st_size / 1e6:.1f} MB")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        convert(Path(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
