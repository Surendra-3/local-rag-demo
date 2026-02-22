import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

df = pd.read_csv("documents.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "people.index")

with open("people_texts.pkl", "wb") as f:
    pickle.dump(df["text"].tolist(), f)
