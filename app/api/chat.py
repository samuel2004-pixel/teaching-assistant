from fastapi import APIRouter
from pydantic import BaseModel

from app.knowledge.retriever import retrieve_context
from app.llm.llm_service import refine_answer
from app.media.tts_engine import text_to_speech

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
def chat(request: ChatRequest):
    # 1. Retrieve context from knowledge base
    raw_context = retrieve_context(request.query)

    # 2. Generate accurate answer using LLM
    final_answer = refine_answer(
        question=request.query,
        context=raw_context
    )

    # 3. ADD NATURAL BREATHING PAUSES (KEY STEP)
    # This makes the voice slow, calm, and teacher-like
    paused_answer = (
        final_answer
        .replace(". ", ".\n\n")   # pause after sentences
        .replace("?", "?\n\n")    # pause after questions
        .replace("!", "!\n\n")    # pause after emphasis
    )

    # 4. Convert answer to audio
    audio_path = text_to_speech(paused_answer)

    # 5. Return response
    return {
        "question": request.query,
        "answer": final_answer,
        "audio_file": audio_path
    }
