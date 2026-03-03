"""
app.py - FastAPI production layer for HR Assistant RAG System.
"""

import os
import logging
import time
import uuid

# Force Hugging Face Hub and Transformers to run entirely offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Import the existing RAG system (unchanged) ──────────────────────────────
from rag_system import HRAssistantRAG, RAGConfig

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hr_api")


# ── Application state ────────────────────────────────────────────────────────
class AppState:
    rag: Optional[HRAssistantRAG] = None
    ready: bool = False
    startup_error: Optional[str] = None


app_state = AppState()


# ── Lifespan: single initialisation at startup ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the RAG system once; tear down on shutdown."""
    logger.info("Starting HR Assistant API — loading RAG system…")
    try:
        config = RAGConfig(
        faiss_index_path="./index/faiss_index.bin",
        metadata_path="./index/chunk_metadata.pkl",
        )
        app_state.rag = HRAssistantRAG(config)
        app_state.ready = True
        logger.info("RAG system loaded successfully.")
    except Exception as exc:
        app_state.startup_error = str(exc)
        logger.error(f"RAG system failed to load: {exc}", exc_info=True)
        # Service starts but /health will report unhealthy

    yield  # ← application runs here

    logger.info("Shutting down HR Assistant API.")
    app_state.rag = None
    app_state.ready = False


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Assistant API",
    description="Production RAG-based HR knowledge assistant.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Employee question to the HR assistant.",
        examples=["How do I apply for sick leave?"],
    )


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(description="Top similarity score [0–1] from retrieval.")
    latency_ms: float
    query_id: str


class HealthResponse(BaseModel):
    status: str                       # "healthy" | "degraded" | "unhealthy"
    rag_loaded: bool
    ollama_reachable: bool
    detail: Optional[str] = None


# ── Middleware: request-ID logging ────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(f"[{request_id}] {response.status_code} — {elapsed:.1f}ms")
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the HR assistant a question.",
    status_code=status.HTTP_200_OK,
)
async def ask(body: AskRequest) -> AskResponse:
    """
    Submit an employee question to the RAG pipeline.

    Returns the generated answer, confidence score, and latency.
    """
    if not app_state.ready or app_state.rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not ready. Please try again shortly.",
        )

    query_id = f"q_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    try:
        answer, metrics = app_state.rag.answer(body.question, query_id=query_id)
    except ValueError as exc:
        # Query validation errors (empty, too long, injection)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # LLM failures after retries
        logger.error(f"[{query_id}] LLM generation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service unavailable. Please try again later.",
        )
    except Exception as exc:
        logger.error(f"[{query_id}] Unexpected pipeline error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )

    return AskResponse(
        answer=answer,
        confidence=round(metrics.top_similarity_score, 4),
        latency_ms=round(metrics.total_time_ms, 2),
        query_id=query_id,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check.",
    status_code=status.HTTP_200_OK,
)
async def health() -> HealthResponse:
    """
    Returns the health of the RAG system and downstream services.

    - `rag_loaded`: FAISS index + models loaded successfully.
    - `ollama_reachable`: Ollama LLM service is reachable.
    """
    rag_loaded = app_state.ready and app_state.rag is not None

    ollama_reachable = False
    if rag_loaded:
        try:
            ollama_reachable = app_state.rag.generator.health_check()
        except Exception:
            ollama_reachable = False

    if rag_loaded and ollama_reachable:
        overall = "healthy"
        http_status = status.HTTP_200_OK
    elif rag_loaded and not ollama_reachable:
        overall = "degraded"
        http_status = status.HTTP_200_OK          # service is up, LLM is down
    else:
        overall = "unhealthy"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    response = HealthResponse(
        status=overall,
        rag_loaded=rag_loaded,
        ollama_reachable=ollama_reachable,
        detail=app_state.startup_error,
    )

    return JSONResponse(content=response.model_dump(), status_code=http_status)


# ── Local dev entry point (do NOT use in production) ─────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
