"""Extract mmap'd .npy sidecars from a token .npz so training reads it with bounded RAM.

Run this ONCE per token .npz produced by ``scripts/export_tokens.py``. It streams each
array straight from the compressed zip member into a memory-mapped .npy sidecar in a
sibling ``<stem>_npy/`` directory, so peak RAM stays bounded even for the 54 GB ``large``
indices array.

Usage:
    pixi run python scripts/extract_token_sidecars.py /path/to/tokens.npz [--overwrite]
"""

import argparse
import logging

from heptokens.data.token_npz import extract_npy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", help="Path to the token .npz file.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract even if sidecars exist."
    )
    parser.add_argument(
        "--row-chunk", type=int, default=200_000, help="Rows streamed per write."
    )
    args = parser.parse_args()
    out_dir = extract_npy(args.npz_path, overwrite=args.overwrite, row_chunk=args.row_chunk)
    print(f"sidecars ready: {out_dir}")


if __name__ == "__main__":
    main()
