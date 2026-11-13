"""Regenerate data/frere2023_nitrocefin/ from the downloaded Zenodo archive.

    python scripts/prepare_frere2023.py [--zip data/raw/frere2023.zip]

The archive is 136 MB, so it is not in git (.gitignore excludes data/raw/*).
Download it with:

    curl -L -o data/raw/frere2023.zip \
       https://zenodo.org/records/10059024/files/Data.zip

Only the 18 experimental CSVs are extracted, not the 671 simulation
workbooks. The instrument's header is stripped and the two numeric columns
are written plainly. The time column needs care: early values are ordinary
decimals ("0.008333") but later ones use a dot as a thousands separator
("1.008.333", meaning 1.008333 min), so a naive float() turns a 1-minute
trace into a 1008-minute one.
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "data" / "raw" / "frere2023.zip"
OUT_DIR = ROOT / "data" / "frere2023_nitrocefin"
PREFIX = "Data/Experimental_Data/"


def parse_time(tok):
    """Instrument time field -> minutes. Handles '1.008.333' style."""
    tok = tok.strip()
    if tok.count(".") > 1:
        head, *rest = tok.split(".")
        return float(f"{head}.{''.join(rest)}")
    return float(tok)


def extract(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not (name.startswith(PREFIX) and name.endswith(".csv")):
                continue
            lines = z.read(name).decode("utf-8", "replace").splitlines()
            start = next(i for i, l in enumerate(lines) if l.startswith("min;"))

            rows = []
            for line in lines[start + 1:]:
                parts = line.split(";")
                if len(parts) < 2 or not parts[1].strip():
                    continue
                try:
                    t = parse_time(parts[0])
                    a = float(parts[1])
                except ValueError:
                    continue
                rows.append(f"{t:.6f},{a:.6f}")

            if not rows:
                continue
            base = Path(name).name
            (out_dir / base).write_text(
                "time_min,absorbance\n" + "\n".join(rows) + "\n"
            )
            written.append(base)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    if not args.zip.exists():
        print(f"archive not found: {args.zip}\nSee this script's docstring "
              f"for the download command.", file=sys.stderr)
        return 1

    written = extract(args.zip, args.out)
    print(f"wrote {len(written)} traces to {args.out}")
    for name in sorted(written):
        print(" ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
