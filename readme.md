# Local RAG Application (FAISS + SentenceTransformers + Ollama + FastAPI)

This project is a fully local Retrieval-Augmented Generation (RAG) system using:

- FAISS for vector similarity search
- SentenceTransformers for embedding generation
- Ollama for running a local LLM
- FastAPI for serving a query API

The system retrieves relevant chunks from a local FAISS index and uses a locally hosted LLM to generate answers.

---

## 1. Environment Setup

### 1.1 Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 1.2 Install dependencies
```bash
pip install fastapi uvicorn faiss-cpu sentence-transformers numpy requests pickle5
```
### 1.3 Install and verify Ollama

Download and install Ollama from:

https://ollama.com

Verify installation:
```bash
ollama --version
```

### 1.4 Pull a local model

Pull at least one model that will be used by the API.

Example:
```bash
ollama pull llama3
```

Make sure the same model name is used in api.py.


## 2. Local Running – End-to-End

Your repository contains the following important files:

`ingestion_sql.py`

`build_index.py` – builds the vector index

`api.py` – FastAPI server

`rag_query.py` – optional CLI test script

The FAISS files are generated automatically:

`people.index`

`people_texts.pkl`

✅ Process
### Step 1 – Build the vector index

You must run your ingestion_sql
`python ingestion_sql.py`

 and existing index builder.

`python build_index.py`

This creates:

`people.index`

`people_texts.pkl`

You must not create these files manually.

### Step 2 – Verify files exist

After Step 1, you must have:

`people.index`
`people_texts.pkl`

Both files must be non-empty as we have fed the system with a query that has non-empty results.

### Step 3 – Start Ollama

In a separate terminal:

`ollama serve`

Verify:

curl http://localhost:11434
### Step 4 – Start the API server
`uvicorn api:app --reload`
### Step 5 – Query the API

Open in browser or curl:

`http://localhost:8000/ask?q=how many products are there?`
### Step 6 – (Optional) CLI RAG test

If you use your standalone script:

`python rag_query.py`
## 3. Architecture
```
DATA SOURCE :: SQL Server (AdventureWorks2025) :: [SQL Server, pyodbc]
        ↓
INGESTION :: [Python, pyodbc, pandas] :: file: Ingest_sql.py
        ↓
TEXT CHUNK / row-to-text preparation :: [Python, pandas] :: file: build_index.py
        ↓
EMBEDING GENERATION :: [sentence-transformers (local, no API)] :: file: build_index.py
        ↓
LOCAL VECTOR DB :: FAISS (local) :: file: build_index.py (writes), people.index, people_texts.pkl
        ↓
SEMANTIC RETRIEVAL :: [FAISS, sentence-transformers, NumPy] :: file: rag_query.py, api.py
        ↓
PROMPT ASSEMBLY :: [Python] :: file: rag_query.py, api.py
        ↓
LOCAL LLM INFERENCE :: [Ollama (HTTP API)] :  file: rag_query.py, api.py
        ↓
RESPONSE/API LAYER :: FastAPI (optional), Python :: file: rag_query.py, api.py

```

One-line pipeline summary
```SQL Server
→ ingest_sql.py
→ build_index.py (text + embeddings)
→ FAISS (people.index + people_texts.pkl)
→ rag_query.py / api.py (retrieval + prompt)
→ Ollama (local LLM)
→ answer
```
### Ingestion & indexing layer

Reads your source data, generates embeddings using Sentence-Transformers, and builds a FAISS index saved as people.index and people_texts.pkl.

### Retrieval layer
FAISS + people.index

At query time, the user question is embedded and searched against the FAISS index to retrieve the most relevant text chunks.

### 3.3 Prompt assembly layer

Takes the retrieved text and builds a constrained prompt:

context only

user question appended

### 3.4 Generation layer

Ollama (local LLM)

Receives the prompt through its HTTP API and generates the final answer locally.

### 3.5 API layer

FastAPI

Exposes /ask endpoint which connects retrieval + generation and returns the final response.

## 4. Issues I faced and how I resolved them
### ❌ Issue 1
could not open people.index for reading

Cause

The index file was missing.

Fix

Run:

python build_index.py

Do not create people.index manually.

### ❌ Issue 2
read error in people.index: 0 != 1

Cause

people.index and people_texts.pkl were created manually and were empty.

Fix

Delete the files and rebuild them properly:

del people.index
del people_texts.pkl
python build_index.py
### ❌ Issue 3
openai.AuthenticationError: Incorrect API key

and / or

no credits / insufficient balance

Cause

The project was still using OpenAI client while the intention was to use a local Ollama model.

Fix

Remove OpenAI usage and switch the API call to Ollama’s local HTTP endpoint.

### ❌ Issue 4
404 Client Error: Not Found for url:
http://localhost:11434/api/generate

Cause

Ollama was not running or the endpoint was incorrect for the request format.

Fix

Start Ollama:

ollama serve

Ensure your API uses the correct Ollama endpoint and payload format.



### ❌ Issue 5

FastAPI shows:

Internal Server Error

and uvicorn logs show:

requests.exceptions.HTTPError

Cause

The Ollama call failed (404 or connection issue), and the exception was not handled.

Fix

Verify:

Ollama is running

model name exists locally

endpoint is correct

Test directly:

curl http://localhost:11434/api/tags
### ❌ Issue 6

I restarted Ollama instead of rebuilding the index

Cause

The error was related to missing FAISS files, not the LLM server.

Fix

Always remember:

FAISS errors → rebuild index

Ollama errors → check Ollama server and API endpoint

## 5. Important operational notes

build_index.py is the only place that creates:

people.index

people_texts.pkl

api.py and rag_query.py only read those files.

Never create index files manually.

Ollama does not require any API key.

OpenAI SDK should not be imported at all when using Ollama.

## 6. Typical startup checklist

✅ Run python build_index.py

✅ Confirm people.index and people_texts.pkl exist

✅ Start Ollama

✅ Start FastAPI

✅ Call /ask

This order avoids almost all of the errors you encountered.