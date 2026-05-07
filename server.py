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

import asyncio
import io
import sys
import os
import logging

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.core.normalizer import normalize_upload
from pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LLM backend selection ─────────────────────────────────────────────────────
# Override with LLM_BACKEND=hf for local dev (no vLLM needed).
# In production on the A100 node, LLM_BACKEND defaults to vllm.
_LLM_BACKEND   = os.environ.get("LLM_BACKEND", "vllm")
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
    Runs the full four-stage pipeline and streams output in two phases:

    Phase 1 — Live feed: each print() from the pipeline is forwarded immediately.
    Phase 2 — Structured output: after pipeline completes, writes %%END_STREAM%%
              followed by the full structured result (PROCESS DETAILS + AUGMENTED
              CONTEXT + REASONING + FINAL REPORT sections).

    NOTE: sys.stdout redirect is not thread-safe under multi-worker deployments.
    Safe for current SLURM single-worker setup. For multi-worker, use
    a threading.local() stdout redirect or a logging handler instead.
    """
    raw = await csv_file.read()

    try:
        variants = normalize_upload(raw, filename=csv_file.filename or "upload.csv")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not variants:
        raise HTTPException(status_code=400, detail="CSV contained no variant rows.")

    pipeline = _get_pipeline()

    # Per-request queue — avoids global stdout contamination in concurrent requests
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    class _QueueWriter(io.TextIOBase):
        """Captures print() output and puts lines into the async queue."""
        def __init__(self):
            self._buf = ""

        def write(self, text: str) -> int:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                loop.call_soon_threadsafe(queue.put_nowait, line + "\n")
            return len(text)

        def flush(self):
            if self._buf:
                loop.call_soon_threadsafe(queue.put_nowait, self._buf)
                self._buf = ""

    async def _run_pipeline():
        """Run the pipeline, redirecting its stdout to the queue."""
        original_stdout = sys.stdout
        writer = _QueueWriter()
        # NOTE: sys.stdout redirect is not thread-safe under multi-worker deployments.
        # Safe for current SLURM single-worker setup. For multi-worker, use
        # a threading.local() stdout redirect or a logging handler instead.
        sys.stdout = writer
        result = None
        error  = None
        try:
            result = await pipeline.run(variants, patient_report)
        except Exception as e:
            error = str(e)
            logger.exception("[Server] Pipeline error during streaming run")
        finally:
            writer.flush()
            sys.stdout = original_stdout
        # Signal end of live feed and emit structured output
        if error:
            await queue.put(f"\n%%END_STREAM%%\n[Pipeline error: {error}]\n")
        else:
            await queue.put(f"\n%%END_STREAM%%\n{result}")
        await queue.put(None)  # sentinel — generator stops

    async def _generate():
        task = asyncio.create_task(_run_pipeline())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await task  # ensure any exceptions are propagated

    return StreamingResponse(
        _generate(),
        media_type="text/plain",
        headers={"X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
