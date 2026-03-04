# RAG AI Teaching Assistant

This project is a **Retrieval-Augmented Generation (RAG) based AI Teaching Assistant** that allows users to ask questions about video lecture content.
The system processes videos, extracts transcripts, creates embeddings, and retrieves relevant information to generate answers using a local LLM.

---

## Features

* Convert lecture videos into audio
* Generate transcripts from audio
* Create semantic embeddings from transcript chunks
* Retrieve relevant lecture segments using similarity search
* Generate answers using a local LLM (Ollama)
* Simple web interface for asking questions

---

## Project Workflow
Videos → Audio → JSON Transcripts → Embeddings → Retrieval → LLM Response

## How to Run
### 1. Add Videos
Place your lecture videos inside the `videos` folder.

### 2. Convert Videos to Audio
Extract audio from videos:
python video_to_mp3.py
Audio files will be saved in the `audio` folder.

### 3. Generate JSON Transcripts
```
python mp3_to_json.py
```
This converts audio files into transcript JSON files stored in `jsons/`.

### 4. Generate Embeddings
```
python preprocess_json.py
```
This creates semantic embeddings and stores them in `embeddings.joblib`.

### 5. Run the Application
```
python main.py
```

Open `index.html` to interact with the system.
---

## Tech Stack
* Python
* SentenceTransformers
* Cosine Similarity Search
* Ollama (Llama 3.2)
* HTML / JavaScript
* Joblib

---

## Project Structure

```
videos/              # Input lecture videos
audio/               # Extracted audio files
jsons/               # Transcript JSON files

video_to_mp3.py      # Video → audio conversion
mp3_to_json.py       # Audio → transcript JSON
preprocess_json.py   # Generate embeddings
process_incoming.py  # RAG query processing
main.py              # Backend server
index.html           # Frontend interface
```

---

## Future Improvements

* Deploy on AWS
* Add vector database (FAISS / Pinecone)
* Improve UI
* Support more document formats
