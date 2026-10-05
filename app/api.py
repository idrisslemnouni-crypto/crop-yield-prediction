"""Run with python -m uvicorn app.api:app --host 127.0.0.1 --port 8000."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, create_model

from crop_yield.predict import load_artifact, predict_rows

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    f"{name}_{month:02d}"
    for month in (4, 5, 6, 7)
    for name in ("tavg", "tmax", "tmin", "prec", "et0", "balance")
]
PredictionRow = create_model(
    "PredictionRow",
    __config__=ConfigDict(extra="forbid", strict=True),
    COUNTY_ID=(str, Field(pattern=r"^[A-Z]{2}_[A-Z_]+$", max_length=80)),
    STATE=(Literal["IA", "IL", "IN", "OH", "MN", "WI", "MI", "MO"], ...),
    FYEAR=(StrictInt, Field(ge=2000, le=2018)),
    SM_WHC=(float, Field(gt=0, le=100, allow_inf_nan=False)),
    **{name: (float, Field(allow_inf_nan=False)) for name in FEATURES},
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PredictionRow] = Field(min_length=1, max_length=100)


def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or Path(os.getenv("CROP_YIELD_MODEL", str(ROOT / "models" / "model.joblib")))

    @asynccontextmanager
    async def lifespan(application):
        application.state.artifact = load_artifact(path) if path.exists() else None
        yield

    application = FastAPI(title="Maize yield benchmark", version="0.1.0", lifespan=lifespan)

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "app" / "index.html")

    @application.get("/health")
    def health():
        artifact = application.state.artifact
        return JSONResponse(
            status_code=200 if artifact else 503,
            content={
                "status": "ready" if artifact else "model_unavailable",
                "model": artifact["selected_model"] if artifact else None,
                "training_years": artifact["training_years"] if artifact else None,
                "scope": "Historical demonstration, not an operational forecast service",
            },
        )

    @application.get("/example")
    def example():
        return {
            "rows": [
                json.loads((ROOT / "reports" / "example-input.json").read_text(encoding="utf-8"))
            ]
        }

    @application.post("/predict")
    def predict(request: PredictionRequest):
        artifact = application.state.artifact
        if artifact is None:
            raise HTTPException(503, "Model unavailable; run the training command first")
        try:
            return {
                "predictions": predict_rows(artifact, [row.model_dump() for row in request.rows])
            }
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return application


app = create_app()
