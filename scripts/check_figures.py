#!/usr/bin/env python3
"""Fail only when a figure's PIXELS change, not when its metadata does.

Why this exists. The CI step originally ran `git diff --exit-code -- figures/`
after regenerating. That went red on all three repos for a reason that had
nothing to do with the figures: matplotlib writes a `tEXt` chunk recording its
own version, so a runner with a different matplotlib than the developer produces
a byte-different PNG whose pixels are identical. The log said it plainly -
`Bin 142376 -> 142376 bytes`, same size, different bytes.

A byte comparison therefore tests the matplotlib version, not the figure. This
compares decoded pixel arrays instead: if the pixels match, the working-tree file
is restored to the committed version (so the tree stays clean and the metadata
noise does not accumulate) and the check passes. If the pixels differ, the
figure genuinely drifted from its committed input and the check fails, naming
the files.

Usage:
    python scripts/check_figures.py [--dir figures]
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def head_bytes(rel: str):
    """The committed blob for a path, or None if it is not tracked."""
    p = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                       capture_output=True)
    return p.stdout if p.returncode == 0 else None


def changed_files(figdir: str):
    p = subprocess.run(["git", "status", "--porcelain", "--", figdir], cwd=ROOT,
                       capture_output=True, text=True)
    out = []
    for line in p.stdout.splitlines():
        status, _, path = line[:2], line[2], line[3:]
        if path.endswith(".png"):
            out.append((status.strip(), path))
    return out


def pixels(b):
    im = Image.open(io.BytesIO(b))
    return np.asarray(im.convert("RGBA"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figures")
    args = ap.parse_args()

    changes = changed_files(args.dir)
    if not changes:
        print(f"figures unchanged: every PNG under {args.dir}/ regenerated "
              f"byte-identically.")
        return 0

    real, meta_only = [], []
    for status, rel in changes:
        if status == "D":
            real.append((rel, "deleted by the regeneration"))
            continue
        old = head_bytes(rel)
        if old is None:
            real.append((rel, "new file: not in the commit"))
            continue
        new = (ROOT / rel).read_bytes()
        try:
            same_pixels = pixels(old).shape == pixels(new).shape and \
                np.array_equal(pixels(old), pixels(new))
        except Exception as exc:                                  # noqa: BLE001
            real.append((rel, f"could not decode: {exc}"))
            continue
        if same_pixels:
            # Metadata-only difference: restore so the tree stays clean.
            subprocess.run(["git", "checkout", "--", rel], cwd=ROOT, check=False)
            meta_only.append(rel)
        else:
            real.append((rel, "pixels differ"))

    if meta_only:
        print("metadata-only differences (pixels identical, file restored):")
        for rel in meta_only:
            print(f"  {rel}")

    if real:
        print("\nFIGURES CHANGED. Their committed inputs and their committed "
              "images no longer agree:")
        for rel, why in real:
            print(f"  {rel}: {why}")
        print("\nRe-commit the regenerated PNGs, or the artefact they read has "
              "drifted from the image in the repo.")
        return 1

    print("\nall differences were metadata only; nothing to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
