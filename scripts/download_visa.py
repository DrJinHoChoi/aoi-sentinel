"""Download VisA dataset (Amazon Science, ECCV 2022, CC BY 4.0).

VisA hosts 12 categories incl. four PCB classes. We grab the full archive,
extract under `data/raw/visa/`, then prune to PCB classes if requested.

Usage:
    python scripts/download_visa.py --out data/raw/visa
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/visa")
    p.add_argument("--pcb-only", action="store_true", help="prune non-PCB classes after extract")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    archive = out / "VisA.tar"
    if not archive.exists():
        print(f"downloading VisA → {archive}")
        urlretrieve(VISA_URL, archive)

    print("extracting ...")
    with tarfile.open(archive) as t:
        t.extractall(out)

    if args.pcb_only:
        keep = {"pcb1", "pcb2", "pcb3", "pcb4"}
        for sub in out.iterdir():
            if sub.is_dir() and sub.name not in keep:
                shutil.rmtree(sub)
        print("pruned to PCB classes")
    print("done")


if __name__ == "__main__":
    main()
