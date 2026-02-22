import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import os
import requests
import os

if not os.path.exists("people.index"):
    raise RuntimeError("people.index not found. Run ingest_sql.py first.")

if not os.path.exists("people_texts.pkl"):
    raise RuntimeError("people_texts.pkl not found. Run ingest_sql.py first.")

print("RAG working directory:", os.getcwd())
# Local Ollama LLM inference
def ask_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()
    return r.json()["response"]

# Embedding model..
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and original texts
index = faiss.read_index("people.index")
texts = pickle.load(open("people_texts.pkl", "rb"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Replace OpenAI call with local Ollama
def ask(question):
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb).astype("float32"), 5)

    context = "\n".join([texts[i] for i in I[0]])

    prompt = f"""
You are answering questions using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# Call Ollama local server instead of OpenAI
    resp_text = ask_ollama(prompt, model="llama2") #mistral
    return resp_text

if __name__ == "__main__":
    print(ask("Isn't 'Road-150 Red, 62' red, why have you not gave this product when I asked 'List all products red'"))
