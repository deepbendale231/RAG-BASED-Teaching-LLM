import os
import json
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

#CONFIG
JSON_DIR = "jsons"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 2000


# LOAD MODEL
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")


def create_embedding(text_list):
    return model.encode(
        text_list,
        normalize_embeddings=True,
        show_progress_bar=False
    )


#CREATE EMBEDDINGS
my_dicts = []
chunk_id = 0

for json_file in os.listdir(JSON_DIR):
    if not json_file.endswith(".json"):
        continue

    with open(
        os.path.join(JSON_DIR, json_file),
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        content = json.load(f)

    print(f"Creating embeddings for {json_file}")

    texts = []
    valid_chunks = []

    for chunk in content["chunks"]:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        texts.append(text[:MAX_CHARS])
        valid_chunks.append(chunk)

    embeddings = create_embedding(texts)

    for i, chunk in enumerate(valid_chunks):
        my_dicts.append({
    "chunk_id": chunk_id,
    "title": chunk.get("title") or chunk.get("video_id"),
    "number": chunk.get("number"),
    "text": chunk.get("text"),
    "start": chunk.get("start"),
    "end": chunk.get("end"),
    "embedding": embeddings[i]
})      
    chunk_id += 1

    


# ---------- DATAFRAME ----------
df = pd.DataFrame.from_records(my_dicts)

print("\n✅ EMBEDDINGS CREATED")
print(f"[{len(df)} rows x {len(df.columns)} columns]")


# ---------- SAVE ----------
joblib.dump(df, "embeddings.joblib")
print(" Saved embeddings to embeddings.joblib")
