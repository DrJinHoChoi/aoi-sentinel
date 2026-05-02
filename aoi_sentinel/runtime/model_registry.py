"""Versioned model registry with atomic swap.

The trainer writes new candidate models here. The edge daemon reads via
`current()` — which always returns a coherent (weights, config, version)
triple. Swap is atomic: prefer a symlink rename (Linux); fall back to a
plain `current.txt` containing the version name (Windows / non-privileged).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ModelHandle:
    version: str
    weights_path: str
    config_path: str
    metadata: dict


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self.current_link = self.root / "current"
        self.current_marker = self.root / "current.txt"  # Windows fallback

    # ----------------------------------------------------------------- write

    def stage(
        self,
        version: str,
        weights_src: str | Path,
        config_src: str | Path,
        metadata: dict | None = None,
    ) -> ModelHandle:
        """Copy artifacts into the registry under a new version directory.

        Does NOT swap `current` — call `promote(version)` once safety_gate clears.
        """
        target = self.root / "versions" / version
        if target.exists():
            raise FileExistsError(target)
        target.mkdir(parents=True)
        shutil.copy2(weights_src, target / "weights.pt")
        shutil.copy2(config_src, target / "config.yaml")
        meta = {
            "version": version,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "promoted": False,
            **(metadata or {}),
        }
        (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ModelHandle(
            version=version,
            weights_path=str(target / "weights.pt"),
            config_path=str(target / "config.yaml"),
            metadata=meta,
        )

    def promote(self, version: str) -> ModelHandle:
        """Atomic swap of `current` → `versions/<version>`.

        Linux: symlink rename (truly atomic, readers never blocked).
        Windows / non-privileged: write `current.txt` containing the version
        (effectively atomic on POSIX-like filesystems via os.replace).
        """
        target = self.root / "versions" / version
        if not target.exists():
            raise FileNotFoundError(target)

        try:
            tmp = self.root / "current.tmp"
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
            os.symlink(Path("versions") / version, tmp)
            os.replace(tmp, self.current_link)
        except (OSError, NotImplementedError):
            # Fallback: write the version into current.txt atomically.
            tmp_marker = self.current_marker.with_suffix(".tmp")
            tmp_marker.write_text(version, encoding="utf-8")
            os.replace(tmp_marker, self.current_marker)

        meta_path = target / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["promoted"] = True
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return ModelHandle(
            version=version,
            weights_path=str(target / "weights.pt"),
            config_path=str(target / "config.yaml"),
            metadata=meta,
        )

    # ------------------------------------------------------------------ read

    def current(self) -> ModelHandle | None:
        target: Path | None = None
        if self.current_link.exists() or self.current_link.is_symlink():
            try:
                target = self.current_link.resolve(strict=True)
            except OSError:
                target = None
        if target is None and self.current_marker.exists():
            version = self.current_marker.read_text(encoding="utf-8").strip()
            cand = self.root / "versions" / version
            if cand.exists():
                target = cand
        if target is None:
            return None
        meta = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        return ModelHandle(
            version=meta["version"],
            weights_path=str(target / "weights.pt"),
            config_path=str(target / "config.yaml"),
            metadata=meta,
        )

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in (self.root / "versions").iterdir() if p.is_dir())

    def rollback(self, to_version: str) -> ModelHandle:
        """Emergency: revert `current` to a known-good prior version."""
        return self.promote(to_version)
