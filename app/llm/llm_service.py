import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def refine_answer(question: str, context: str) -> str:
    prompt = f"""
You are an expert engineering professor.

RULES (STRICT):
- Answer ONLY what is asked in the question
- Use ONLY the provided context
- Do NOT add extra topics
- Do NOT give future scope, advantages, or examples unless asked
- Keep the answer concise (max 6–8 sentences)
- Stop immediately after answering the question

Context:
{context}

Question:
{question}

Give a focused, exam-ready explanation.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return context
