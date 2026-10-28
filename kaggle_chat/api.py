"""MedRAG-SLM Community Health Worker Standalone Backend API.

FastAPI service exposing health checks, emergency triage screening,
and conversational clinical decision support with calibrated safety abstention.

To run locally in mock mode:
    MOCK=1 python3 -m uvicorn kaggle_chat.api:app --port 8000
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from kaggle_chat.red_flags import (
    EMERGENCY_PREFIX,
    MANDATORY_FOOTER,
    check_emergency,
)

__all__ = [
    "app",
    "check_emergency",
    "EMERGENCY_PREFIX",
    "MANDATORY_FOOTER",
    "DISCLAIMER",
]

# -----------------------------------------------------------------------------
# Clinical Disclaimer & Canned Messages
# -----------------------------------------------------------------------------
DISCLAIMER: str = (
    "This tool supports, and does not replace, clinical judgement. "
    "Follow local protocols and refer urgent cases."
)

ABSTAIN_DEFAULT_MSG: str = (
    "I don't have enough information to answer this confidently. Please consult a doctor."
)

try:
    from src.llm import ABSTAIN_MESSAGE as _SRC_ABSTAIN_MSG

    _ABSTAIN_MESSAGE: str = _SRC_ABSTAIN_MSG
except Exception:
    _ABSTAIN_MESSAGE = ABSTAIN_DEFAULT_MSG


def is_mock_mode() -> bool:
    """Return True if environment variable MOCK is set to 1."""
    return os.environ.get("MOCK", "").strip() == "1"


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str
    model: str = "medpsy-4b"
    model_loaded: bool = False
    disclaimer: str = DISCLAIMER


class EmergencyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class EmergencyResponse(BaseModel):
    is_emergency: bool
    reason: Optional[str] = None
    result: Tuple[bool, Optional[str]]
    emergency_checker: str = "red_flags"
    disclaimer: str = DISCLAIMER


class Citation(BaseModel):
    id: Union[int, str]
    text: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    red_flag: bool
    emergency_reason: Optional[str] = None
    emergency_checker: str = "red_flags"
    abstained: bool
    citations: List[Citation]
    latency_s: float
    mode: str
    disclaimer: str = DISCLAIMER


# -----------------------------------------------------------------------------
# FastAPI Application & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="MedRAG-SLM Clinical Support API",
    version="1.0.0",
    description="Backend service providing clinical triage and decision support.",
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next: Any) -> Response:
    """Attach X-Request-ID and Request-ID headers to every HTTP response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Request-ID"] = request_id
    return response


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """System liveness and readiness probe."""
    mode = "mock" if is_mock_mode() else "live"
    return HealthResponse(
        status="ok",
        mode=mode,
        model="medpsy-4b",
        model_loaded=False,
        disclaimer=DISCLAIMER,
    )


@app.post("/api/v1/emergency", response_model=EmergencyResponse)
def post_emergency(req: EmergencyRequest) -> EmergencyResponse:
    """Screen a query for acute red-flag medical emergencies."""
    if not req.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace only.",
        )

    res = check_emergency(req.query)
    is_emergency, reason = res

    return EmergencyResponse(
        is_emergency=is_emergency,
        reason=reason,
        result=res,
        emergency_checker="red_flags",
        disclaimer=DISCLAIMER,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest) -> ChatResponse:
    """Clinical dialogue endpoint with emergency screening and mock mode."""
    t_start = time.perf_counter()

    if not req.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace only.",
        )

    # Emergency check ALWAYS runs first before model or mock evaluation
    is_emergency, reason = check_emergency(req.query)
    if is_emergency:
        return ChatResponse(
            session_id=req.session_id,
            answer=EMERGENCY_PREFIX.strip(),
            red_flag=True,
            emergency_reason=reason,
            emergency_checker="red_flags",
            abstained=False,
            citations=[],
            latency_s=round(time.perf_counter() - t_start, 4),
            mode="mock" if is_mock_mode() else "live",
            disclaimer=DISCLAIMER,
        )

    # Live mode is not yet implemented
    if not is_mock_mode():
        raise HTTPException(
            status_code=501,
            detail="Live model inference is not yet implemented. Set MOCK=1 to run in mock mode.",
        )

    # Mock mode handling
    if "unanswerable" in req.query.lower():
        return ChatResponse(
            session_id=req.session_id,
            answer=_ABSTAIN_MESSAGE,
            red_flag=False,
            emergency_reason=None,
            emergency_checker="red_flags",
            abstained=True,
            citations=[],
            latency_s=round(time.perf_counter() - t_start, 4),
            mode="mock",
            disclaimer=DISCLAIMER,
        )

    mock_citations = [
        Citation(
            id="chunk_pediatrics_01",
            text="Nelson Textbook of Pediatrics, 21st ed.: Evaluation of acute pediatric symptoms and hydration management.",
        ),
        Citation(
            id="chunk_internal_med_02",
            text="Harrison's Principles of Internal Medicine, 21st ed.: Clinical decision making in outpatient primary care.",
        ),
    ]

    mock_answer = (
        "Based on clinical guidance, assess the patient's hydration, monitor temperature and "
        "vital signs, and refer to a primary health centre if symptoms worsen or do not resolve within 48 hours."
    )

    return ChatResponse(
        session_id=req.session_id,
        answer=mock_answer,
        red_flag=False,
        emergency_reason=None,
        emergency_checker="red_flags",
        abstained=False,
        citations=mock_citations,
        latency_s=round(time.perf_counter() - t_start, 4),
        mode="mock",
        disclaimer=DISCLAIMER,
    )
