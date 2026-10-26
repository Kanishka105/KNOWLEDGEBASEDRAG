
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4
import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.conversation_store import ConversationStore

from app.ai.rag.phase4_rag_generation.rag_pipeline import (
    GEMINI_MODEL,
    create_gemini_client,
    load_retrieval_components,
    answer_question as run_rag_question,
)
from app.ai.rag.phase6_document_ingestion.config import UPLOAD_DIR
from app.ai.rag.phase6_document_ingestion.document_store import DocumentStore
from app.ai.rag.phase6_document_ingestion.pdf_qa import (
    answer_from_uploaded_pdfs,
)

from app.ai.rag.phase10_agentic_research.agent import (
    run_agentic_research,
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

configure_logging()

logger = logging.getLogger(__name__)
settings = get_settings()
MAX_UPLOAD_BYTES = settings.max_upload_bytes


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_ready = False
    app.state.documents_ready = False
    app.state.gemini_client = None
    app.state.rag_components = None
    app.state.document_store = None
    app.state.conversation_store = None

    try:
        logger.info("Loading SciFact RAG components...")

        app.state.gemini_client = create_gemini_client()
        app.state.rag_components = load_retrieval_components()
        app.state.rag_ready = True

        logger.info("Loading uploaded-document store...")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        app.state.document_store = DocumentStore()
        app.state.documents_ready = True

        logger.info("Loading conversation store...")

        app.state.conversation_store = ConversationStore()

        logger.info("RAG API ready.")
        yield

    finally:
        app.state.rag_ready = False
        app.state.documents_ready = False
        logger.info("RAG API shutting down.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Scientific research RAG, agentic research, "
        "uploaded-PDF question answering, and conversation history."
    ),
    version="1.4.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST AND RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SourceResponse(BaseModel):
    source_label: str
    document_id: str
    title: str
    retrieval_rank: int | None = None
    evidence: str


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    retrieval_method: str
    sources: list[SourceResponse]
    citation_validation: dict[str, Any]


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class DocumentAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=3)


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=200)


class AddMessageRequest(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=20000)
    sources: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================
# SHARED HELPERS
# ============================================================

def get_document_store(request: Request) -> DocumentStore:
    if not getattr(request.app.state, "documents_ready", False):
        raise HTTPException(
            status_code=503,
            detail="Uploaded-document store is not ready.",
        )

    store = getattr(request.app.state, "document_store", None)

    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Uploaded-document store is not initialized.",
        )

    return store


def get_gemini_client(request: Request):
    client = getattr(request.app.state, "gemini_client", None)

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini client is not initialized.",
        )

    return client


def get_conversation_store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conversation_store", None)

    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation store is not initialized.",
        )

    return store


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    error_type = type(exc).__name__.lower()

    return any(
        marker in message or marker in error_type
        for marker in (
            "resource_exhausted",
            "quota",
            "too many requests",
            "429",
        )
    )
def resolve_followup_question(
    question: str,
    messages: list[dict],
    client,
) -> str:
    """Rewrite a follow-up question using recent conversation history."""

    recent_messages = messages[-8:]

    if not recent_messages:
        return question.strip()

    history = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_messages
    )

    prompt = f"""
You are a scientific research question-rewriting assistant.

Use the conversation history to understand the latest question.
Rewrite it as a standalone research question.

Rules:
- Resolve references such as "it", "they", and "that study".
- Preserve the user's original intent.
- Do not answer the question.
- Do not add unsupported scientific claims.
- Return only the rewritten question.
- If the question is already clear, return it unchanged.

CONVERSATION HISTORY:
{history}

LATEST QUESTION:
{question}
""".strip()

    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    rewritten = (response.output_text or "").strip()

    if not rewritten:
        raise RuntimeError(
            "The model returned an empty rewritten question."
        )

    return rewritten

def is_provider_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    error_type = type(exc).__name__.lower()

    return any(
        marker in message or marker in error_type
        for marker in (
            "service_unavailable",
            "high demand",
            "503",
            "temporarily unavailable",
            "internalservererror",
        )
    )


def raise_provider_http_error(exc: Exception) -> None:
    if is_quota_error(exc):
        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini API quota or rate limit has been reached. "
                "Please try again after the limit resets."
            ),
        ) from exc

    if is_provider_unavailable_error(exc):
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable or experiencing "
                "high demand. Please retry in a few minutes."
            ),
        ) from exc


# ============================================================
# ROOT AND HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "application": settings.app_name,
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "scientific_rag": "/api/v1/ask",
            "agentic_research": "/api/v1/agentic-research",
            "upload_pdf": "/api/v1/documents/upload",
            "list_documents": "/api/v1/documents",
            "search_documents": "/api/v1/documents/search",
            "ask_uploaded_documents": "/api/v1/documents/ask",
            "create_conversation": "/api/v1/conversations",
            "list_conversations": "/api/v1/conversations",
        },
    }


@app.get("/health")
def health(request: Request):
    rag_ready = getattr(request.app.state, "rag_ready", False)
    documents_ready = getattr(
        request.app.state,
        "documents_ready",
        False,
    )
    conversations_ready = (
        getattr(request.app.state, "conversation_store", None) is not None
    )

    ready = rag_ready and documents_ready and conversations_ready

    return {
        "status": "ok" if ready else "starting",
        "rag_ready": rag_ready,
        "documents_ready": documents_ready,
        "conversations_ready": conversations_ready,
    }


# ============================================================
# SCIENTIFIC RAG QUESTION ANSWERING
# ============================================================

@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(body: AskRequest, request: Request):
    if not getattr(request.app.state, "rag_ready", False):
        raise HTTPException(
            status_code=503,
            detail="SciFact RAG components are not ready.",
        )

    try:
        return run_rag_question(
            question=body.question.strip(),
            components=request.app.state.rag_components,
            client=get_gemini_client(request),
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("SciFact RAG failed.")
        raise_provider_http_error(exc)
        raise HTTPException(
            status_code=500,
            detail="SciFact RAG failed. Check server logs.",
        ) from exc


# ============================================================
# PHASE 10 — AGENTIC RESEARCH
# ============================================================

@app.post("/api/v1/agentic-research")
def agentic_research(body: AskRequest, request: Request):
    if not getattr(request.app.state, "rag_ready", False):
        raise HTTPException(
            status_code=503,
            detail="SciFact RAG components are not ready.",
        )

    try:
        return run_agentic_research(
            question=body.question.strip(),
            components=request.app.state.rag_components,
            client=get_gemini_client(request),
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Agentic research workflow failed.")
        raise_provider_http_error(exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during agentic research.",
        ) from exc


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/api/v1/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    store = get_document_store(request)
    original_name = Path(file.filename or "").name

    if (
        not original_name
        or Path(original_name).suffix.lower() != ".pdf"
    ):
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload a file with a .pdf extension.",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {limit_mb:g} MB upload limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has an invalid PDF signature.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}_{original_name}"
    destination = UPLOAD_DIR / stored_name

    try:
        destination.write_bytes(content)
        result = store.ingest_pdf(destination)
        result["original_filename"] = original_name
        return result

    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("PDF ingestion failed.")
        raise HTTPException(
            status_code=500,
            detail="PDF ingestion failed. Check server logs.",
        ) from exc


# ============================================================
# LIST UPLOADED DOCUMENTS
# ============================================================

@app.get("/api/v1/documents")
def list_documents(request: Request):
    store = get_document_store(request)

    return {
        "count": len(store.documents),
        "documents": store.documents,
    }


# ============================================================
# SEARCH UPLOADED PDF EVIDENCE
# ============================================================
@app.post(
    "/api/v1/conversations/{conversation_id}/messages",
    status_code=201,
)
@app.post("/api/v1/documents/search")
def search_documents(
    body: DocumentSearchRequest,
    request: Request,
):
    store = get_document_store(request)

    try:
        results = store.search(
            query=body.query.strip(),
            top_k=body.top_k,
        )

        return {
            "query": body.query,
            "count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Document search failed.")
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Check server logs.",
        ) from exc


# ============================================================
# QUESTION ANSWERING OVER UPLOADED PDFs
# ============================================================

@app.post("/api/v1/documents/ask")
def ask_uploaded_documents(
    body: DocumentAskRequest,
    request: Request,
):
    store = get_document_store(request)

    try:
        return answer_from_uploaded_pdfs(
            question=body.question.strip(),
            document_store=store,
            client=get_gemini_client(request),
            top_k=body.top_k,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Uploaded-document QA failed.")
        raise_provider_http_error(exc)
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to answer from uploaded PDFs. "
                "Check the server terminal for the error."
            ),
        ) from exc


# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

@app.post("/api/v1/conversations", status_code=201)
def create_conversation(
    body: CreateConversationRequest,
    request: Request,
):
    store = get_conversation_store(request)
    return store.create_conversation(body.title)


@app.get("/api/v1/conversations")
def list_conversations(request: Request):
    store = get_conversation_store(request)
    conversations = store.list_conversations()

    return {
        "count": len(conversations),
        "conversations": conversations,
    }


@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    request: Request,
):
    store = get_conversation_store(request)
    conversation = store.get_conversation(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


@app.post(
    "/api/v1/conversations/{conversation_id}/messages",
    status_code=201,
)
def add_conversation_message(
    conversation_id: str,
    body: AddMessageRequest,
    request: Request,
):
    store = get_conversation_store(request)

    try:
        message = store.add_message(
            conversation_id=conversation_id,
            role=body.role,
            content=body.content,
            sources=body.sources,
        )

        return {
            "conversation_id": conversation_id,
            "message": message,
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
@app.post("/api/v1/conversations/{conversation_id}/ask")
def ask_in_conversation(
    conversation_id: str,
    body: AskRequest,
    request: Request,
):
    if not getattr(request.app.state, "rag_ready", False):
        raise HTTPException(
            status_code=503,
            detail="SciFact RAG components are not ready.",
        )

    conversation_store = get_conversation_store(request)
    client = get_gemini_client(request)

    conversation = conversation_store.get_conversation(
        conversation_id
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    original_question = body.question.strip()

    try:
        # Load previous history before saving the current question.
        history = conversation["messages"]

        # Rewrite the question using previous messages.
        standalone_question = resolve_followup_question(
            question=original_question,
            messages=history,
            client=client,
        )

        # Retrieve scientific evidence and generate the answer.
        result = run_rag_question(
            question=standalone_question,
            components=request.app.state.rag_components,
            client=client,
        )

        # Preserve the original question in the API response.
        result["question"] = original_question
        result["standalone_question"] = standalone_question
        result["conversation_id"] = conversation_id

        # Save the user's question.
        conversation_store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=original_question,
        )

        # Save the assistant's answer and sources.
        conversation_store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            sources=result.get("sources", []),
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Conversation-aware RAG failed.")
        raise_provider_http_error(exc)

        raise HTTPException(
            status_code=500,
            detail=(
                "Conversation-aware research failed. "
                "Check the server terminal for the error."
            ),
        ) from exc