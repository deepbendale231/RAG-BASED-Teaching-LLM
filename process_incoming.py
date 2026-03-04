import numpy as np
import pandas as pd
import joblib
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"        # ollama model
TOP_RESULTS = 5

# ---------------- LOAD EMBEDDING MODEL ----------------
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)
print("Embedding model loaded.")

def create_embedding(text_list):
    return embed_model.encode(
        text_list,
        normalize_embeddings=True,
        show_progress_bar=False
    )

# ---------------- OLLAMA INFERENCE ----------------
def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"]

# ---------------- LOAD EMBEDDINGS ----------------
df = joblib.load("embeddings.joblib")
print(f"Loaded {len(df)} embedded chunks")

# ---------------- CORE RAG FUNCTION ----------------
def answer_question(incoming_query: str):
    # Embed the question
    question_embedding = create_embedding([incoming_query])[0]

    # Similarity search
    embedding_matrix = np.vstack(df["embedding"].values)

    similarities = cosine_similarity(
        embedding_matrix,
        [question_embedding]
    ).flatten()

    top_indices = similarities.argsort()[::-1][:TOP_RESULTS]
    new_df = df.loc[top_indices]

    # Build context JSON
    context_json = new_df[
        ["title", "number", "start", "end", "text"]
    ].to_dict(orient="records")

    # Prompt construction
    prompt = f"""
I am teaching Python for beginners in my python course. I have a set of videos with subtitles.
Each subtitle chunk is associated with a video title, video number, start time, end time, and the spoken text.

Context:
{json.dumps(context_json, ensure_ascii=False, indent=2)}

---------------------------------

User question:
"{incoming_query}"

Instructions:
- Answer in a clear, friendly, human way
- Explain WHICH video(s) the topic is taught in
- Mention APPROXIMATE timestamps (start–end)
- Guide the user to watch that specific part of the video
- Do NOT mention embeddings, chunks, or JSON
- If the question is NOT related to this course, politely say you can only answer course-related questions
"""

    # LLM response
    response = inference(prompt)

    return {
        "question": incoming_query,
        "answer": response,
        "sources": context_json
    }

# ---------------- CLI TEST (SAFE) ----------------
if __name__ == "__main__":
    result = answer_question("What is a variable in Python and how do I use it?")
    print("\n=== ANSWER ===\n")
    print(result["answer"])
