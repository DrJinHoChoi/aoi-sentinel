"""FastAPI inference server.

Exposes a /classify endpoint that takes an ROI crop and returns
true-defect / false-call probability. More routes added as phases ship.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="aoi-sentinel", version="0.1.0")


class ClassifyResponse(BaseModel):
    is_true_defect: bool
    confidence: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify() -> ClassifyResponse:
    # TODO: load model lazily, accept image upload, return real prediction
    raise NotImplementedError
