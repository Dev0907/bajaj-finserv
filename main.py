from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from helper import load_pdf_file, filter_to_minimal_docs, text_split
from store_index import store_chunks_in_pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import requests
from sentence_transformers import SentenceTransformer

app = FastAPI()
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_pipeline(request_data: QueryRequest):
    # Download PDF
    response = requests.get(request_data.documents)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download PDF")

    os.makedirs("data", exist_ok=True)
    with open("data/input.pdf", "wb") as f:
        f.write(response.content)

    # Load and split document
    docs = load_pdf_file("data/")
    minimal_docs = filter_to_minimal_docs(docs)
    chunks = text_split(minimal_docs)

    # Index into Pinecone only once
    if not os.path.exists("index_built.flag"):
        store_chunks_in_pinecone(chunks)
        with open("index_built.flag", "w") as f:
            f.write("done")

    # Prepare retrieval and LLM inference
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("medical-chatbot")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    answers = []
    for q in request_data.questions:
        query_embedding = model.encode([q], show_progress_bar=False)[0].tolist()
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        retrieved = result.get("matches", [])
        context = "\n\n".join([match["metadata"]["text"][:500] for match in retrieved])

        # Prompt the Groq model
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are an intelligent insurance assistant.\n"
            "Use the following policy context to answer the user's question:\n\n"
            f"{context}\n\n"
            f"Question: {q}\n\n"
            "Respond with a clear, direct answer to the user's question, using verbatim lines from the policy document wherever possible.\n"
            "Keep the answer complete and self-explanatory in 2-3 sentences.\n"
            "If the context does not provide enough details, respond with: Insufficient information."
        )

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an insurance assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }

        groq_response = requests.post(url, headers=headers, json=payload)
        if groq_response.status_code == 200:
            #groq_json = groq_response.json()
            groq_json = groq_response.json()
            raw_content = groq_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            cleaned_content = raw_content.replace("\\", "").strip()
            answers.append(cleaned_content)
        else:
            answers.append("Insufficient information: {groq_response.text}")

    return QueryResponse(answers=answers)
