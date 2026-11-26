import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai

app = FastAPI(title="PDF RAG Chatbot")
PDF_PATH = Path(__file__).parent / "knowledge.pdf.pdf"
MODEL = "gemini-3.1-flash-lite"


def load_chunks(chunk_size=500, overlap=100):
    if not PDF_PATH.exists():
        return []

    reader = PdfReader(str(PDF_PATH))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        return []

    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
        if words[i:i + chunk_size]
    ]


chunks = load_chunks()


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PDF RAG Chatbot</title>
      </head>
      <body style="font-family:Arial,sans-serif;max-width:720px;margin:40px auto;padding:20px">
        <h1>📄 PDF RAG Chatbot</h1>
        <p>Ask questions about the PDF connected to this project.</p>
        <textarea id="question" rows="4" placeholder="Type your question here"
          style="box-sizing:border-box;width:100%;padding:12px"></textarea>
        <button onclick="ask()" style="margin-top:12px;padding:10px 18px">Ask</button>
        <p id="status"></p>
        <div id="answer" style="white-space:pre-wrap;line-height:1.5"></div>
        <script>
          async function ask() {
            const question = document.getElementById("question").value;
            const status = document.getElementById("status");
            const answer = document.getElementById("answer");
            status.textContent = "Thinking...";
            answer.textContent = "";
            try {
              const response = await fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question})
              });
              const data = await response.json();
              if (!response.ok) throw new Error(data.detail || "Request failed");
              answer.textContent = data.answer;
              status.textContent = "";
            } catch (error) {
              status.textContent = error.message;
            }
          }
        </script>
      </body>
    </html>
    """


@app.post("/ask")
def ask(data: Question):
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if not chunks:
        raise HTTPException(
            status_code=503,
            detail="No readable PDF found. Add your own knowledge.pdf.pdf beside app.py."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    # Lightweight keyword-based retrieval keeps this deployment small.
    terms = {word.lower() for word in question.split() if len(word) > 2}
    ranked = sorted(
        chunks,
        key=lambda chunk: sum(term in chunk.lower() for term in terms),
        reverse=True
    )
    context = "\n\n".join(ranked[:3])

    prompt = f"""You are a document question-answering assistant.
Use only the supplied document context.
If the answer is not in the context, say:
"The information is not available in the uploaded document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return {"answer": response.text or "Sorry, I could not generate an answer."}
