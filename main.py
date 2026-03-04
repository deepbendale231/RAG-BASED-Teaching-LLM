from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from process_incoming import answer_question

app = FastAPI()

# 🔥 CORS MUST COME BEFORE ROUTES
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # 👈 THIS enables OPTIONS
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(data: AskRequest):
    return answer_question(data.question)
