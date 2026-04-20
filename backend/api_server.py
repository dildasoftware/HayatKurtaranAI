from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag_engine import generate_answer  # pyre-ignore


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    is_critical: bool
    has_context: bool
    sources: list[dict]


app = FastAPI(title="HayatKurtaran AI API", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = generate_answer(req.query)
    except EnvironmentError as e:
        # API anahtarı gibi ortam değişkenleri eksik olduğunda net hata dön.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Sistem hatası oluştu. Lütfen daha sonra tekrar deneyin.",
        )

    return ChatResponse(
        answer=result.get("answer", ""),
        is_critical=bool(result.get("is_critical", False)),
        has_context=bool(result.get("has_context", True)),
        sources=list(result.get("sources", [])),
    )

