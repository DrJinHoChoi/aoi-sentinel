"""Clone DeepPCB (Tang et al.) — bare PCB defect dataset with template/test pairs.

The dataset is distributed via the GitHub repo's `PCBData/` directory. We do a
shallow clone so we don't pull the entire git history.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = "https://github.com/tangsanli5201/DeepPCB.git"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/deeppcb")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists — skipping clone")
        return
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(out)])
    print(f"DeepPCB ready under {out}")


if __name__ == "__main__":
    main()
