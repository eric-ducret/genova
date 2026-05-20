"""
server.py — FastAPI entry point for the variant analysis pipeline.

Thin server: parses requests, delegates to pipeline.run(), returns results.
No business logic lives here — all logic is in pipeline/.

Routes:
    GET  /health     — liveness check
    POST /analyze    — main pipeline endpoint

The /analyze route signature is fixed and must not be changed:
existing frontends depend on it (MIGRATION.md constraint).

Startup:
    Pipeline is instantiated once at server startup via the FastAPI lifespan.
    LLM client and manifests are loaded at that point, not per request.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.core.normalizer import normalize_upload
from pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LLM backend selection ─────────────────────────────────────────────────────
# Override with LLM_BACKEND=hf for local dev (no vLLM needed).
# In production on the A100 node, LLM_BACKEND defaults to vllm.
_LLM_BACKEND   = os.environ.get("LLM_BACKEND", "vllm")
_RESULTS_DIR   = Path(__file__).parent / "results"
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
_USE_4BIT      = os.environ.get("USE_4BIT", "0") == "1"

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Variant Analysis Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline singleton — instantiated on first request (lazy) or at startup.
# Using a module-level variable avoids a startup dependency on the GPU being ready.
_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("[Server] Initialising Pipeline (backend=%s)", _LLM_BACKEND)
        if _LLM_BACKEND == "hf":
            # Dev / CPU fallback — build HFClient directly and pass it in
            from pipeline.llm.hf_client import HFClient
            _llm = HFClient(model_id="Qwen/Qwen3.5-9B", use_4bit=_USE_4BIT)
            _pipeline = Pipeline(llm=_llm)
        else:
            # Production: vLLM OpenAI-compatible endpoint
            _pipeline = Pipeline(
                model_name="qwen3.5-9b",
                llm_kwargs={"base_url": _VLLM_BASE_URL},
            )
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Returns 200 if the server process is running."""
    return {"status": "ok", "model": "qwen3.5-9b", "backend": _LLM_BACKEND}


@app.post("/analyze")
async def analyze(
    csv_file:       UploadFile = File(..., description="Variants CSV file"),
    patient_report: str        = Form(..., description="Free-text patient phenotype"),
):
    """
    Upload a variants CSV + patient phenotype description.
    Runs the full four-stage pipeline and returns the structured report as plain text.
    """
    raw = await csv_file.read()

    try:
        variants, raw_rows = normalize_upload(
            raw, filename=csv_file.filename or "upload.csv"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not variants:
        raise HTTPException(status_code=400, detail="CSV contained no variant rows.")

    pipeline = _get_pipeline()
    result = await pipeline.run(
        variants,
        patient_report,
        raw_rows=raw_rows,
        # genome_build uses DEFAULT_GENOME_BUILD — auto-detection comes later
    )

    stem = Path(csv_file.filename or "upload").stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _RESULTS_DIR / f"{stem}_{timestamp}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.txt").write_text(result, encoding="utf-8")
        logger.info("[Server] Result saved to %s", run_dir)
    except Exception as e:
        logger.warning("[Server] Could not save result to disk: %s", e)

    return PlainTextResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
