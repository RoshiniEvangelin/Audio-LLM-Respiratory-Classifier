"""
Run this AFTER you've cloned and set up the real OPERA repo
(https://github.com/evelyn0414/OPERA) to precompute OPERA-CT embeddings for
every audio file you have, so training never needs to run the audio encoder
live (it's frozen anyway).

Usage (from inside your activated OPERA conda env, with this repo's
directory on PYTHONPATH):

    python precompute_opera_features.py \
        --audio_list audio_paths.txt \
        --out opera_features.npz

`audio_paths.txt`: one audio file path per line. The sample id used for
lookup is the file's stem (filename without extension) — change
`sample_id_fn` below if you need a different id scheme (e.g. a CSV row id).
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_list", required=True, help="text file, one audio path per line")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--input_sec", type=float, default=8.0)
    args = ap.parse_args()

    # Import lazily — only needed here, and only exists once the OPERA repo
    # is on your PYTHONPATH.
    try:
        from src.benchmark.model_util import extract_opera_feature
    except ImportError as e:
        raise SystemExit(
            "Could not import extract_opera_feature. Clone "
            "https://github.com/evelyn0414/OPERA, follow its README to set "
            "up the 'audio' conda env, and run this script from within that "
            "repo's directory (or add it to PYTHONPATH)."
        ) from e

    paths = [l.strip() for l in Path(args.audio_list).read_text().splitlines() if l.strip()]
    ids = [Path(p).stem for p in paths]

    print(f"[precompute] extracting OPERA-CT features for {len(paths)} files...")
    features = extract_opera_feature(
        np.array(paths), pretrain="operaCT", input_sec=args.input_sec, dim=args.dim
    )
    features = np.array(features)

    np.savez(args.out, embeddings=features, ids=np.array(ids, dtype=object))
    print(f"[precompute] saved {features.shape} embeddings to {args.out}")


if __name__ == "__main__":
    main()
