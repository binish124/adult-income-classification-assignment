"""FastAPI endpoint for Adult Income predictions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"
api = FastAPI(title="Adult Income Classifier", version="1.0.0")


class AdultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=120)
    education_num: int | None = Field(default=None, ge=1, le=16)
    capital_gain: int | None = Field(default=None, ge=0)
    capital_loss: int | None = Field(default=None, ge=0)
    hours_per_week: int | None = Field(default=None, ge=1, le=99)
    workclass: str | None = None
    marital_status: str | None = None
    occupation: str | None = None


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python train.py` to create models/model.joblib")
    return joblib.load(MODEL_PATH)


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@api.post("/predict")
def predict(record: AdultRecord) -> dict:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    row = pd.DataFrame([record.model_dump()]).replace({"?": pd.NA, "": pd.NA})
    for col in ("workclass", "marital_status", "occupation"):
        row[col] = row[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
        row[col] = row[col].astype(object).where(row[col].notna(), np.nan)
    probability = float(model.predict_proba(row)[0, 1])
    label = ">50K" if probability >= 0.5 else "<=50K"
    return {"prediction": label, "probability_over_50k": round(probability, 4)}
