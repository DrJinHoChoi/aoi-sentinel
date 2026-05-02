"""Operator web UI (FastAPI + HTMX).

One screen, three modes:

  SHADOW       — every flagged ROI shown; PASS/DEFECT/UNSURE buttons.
  ASSIST       — only disagreements shown; engine pre-fills its choice.
  AUTONOMOUS   — KPI dashboard only; no per-ROI interaction.

Mobile-first card layout — most lines have a tablet at the workstation, not a
full PC. Swipe right = PASS, swipe left = DEFECT, tap-and-hold = UNSURE.
"""
from aoi_sentinel.ui.web.app import app

__all__ = ["app"]
