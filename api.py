import time
import faiss
import pickle
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(os.path.join(BASE_DIR, "people.index"))
texts = pickle.load(open(os.path.join(BASE_DIR, "people_texts.pkl"), "rb"))


def ask_ollama(prompt: str, model_name: str = "llama2") -> str:
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()

    return r.json()["response"]


@app.get("/ask")
def ask(q: str):

    start = time.time()

    q_emb = model.encode([q])
    D, I = index.search(np.array(q_emb).astype("float32"), 5)

    if len(I[0]) == 0:
        return {"answer": "No relevant data found"}

    context = "\n".join([texts[i] for i in I[0]])

    prompt = f"""
Use only the following context to answer.

Context:
{context}

Question:
{q}

Answer:
"""

    answer = ask_ollama(prompt)

    latency = time.time() - start

    return {
        "answer": answer,
        "latency_seconds": round(latency, 3)
    }


# import time
# import faiss
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI
# #from openai import OpenAI
# import os

# app = FastAPI()

# model = SentenceTransformer("all-MiniLM-L6-v2")
# index = faiss.read_index("people.index")
# texts = pickle.load(open("people_texts.pkl", "rb"))

# #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# @app.get("/ask")
# def ask(q: str):

    # start = time.time()

    # q_emb = model.encode([q])
    # D, I = index.search(np.array(q_emb).astype("float32"), 5)

    # if len(I[0]) == 0:
        # return {"answer": "No relevant data found"}

    # context = "\n".join([texts[i] for i in I[0]])

    # prompt = f"""
# Use only the following context.

# {context}

# Question:
# {q}
# """

    # resp = client.chat.completions.create(
        # model="gpt-4o-mini",
        # messages=[{"role":"user","content":prompt}],
        # temperature=0
    # )

    # latency = time.time() - start

    # return {
        # "answer": resp.choices[0].message.content,
        # "latency_seconds": round(latency, 3)
    # }