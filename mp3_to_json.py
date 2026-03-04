import os
import json
import subprocess
import time
from openai import OpenAI

# ================= CONFIG =================
AUDIO_DIR = "audio"
TEMP_CHUNKS_DIR = "audio_chunks"
OUTPUT_JSON_DIR = "jsons"

MODEL = "gpt-4o-mini-transcribe"

CHUNK_SECONDS = 120      # 2 minutes
MAX_RETRIES = 5
WAIT_SECONDS = 8

client = OpenAI()

os.makedirs(TEMP_CHUNKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


def split_audio(audio_path):    # USED FOR SPLITTING AUDIO INTO CHUNKS
    """Split audio into small chunks using ffmpeg"""
    # clean old chunks
    for f in os.listdir(TEMP_CHUNKS_DIR):
        os.remove(os.path.join(TEMP_CHUNKS_DIR, f))

    cmd = [                                         # IT SAYS: “FFmpeg, take this audio, split it into 2-minute segments, and save them in the temp chunks directory with sequential names.”
        "ffmpeg",
        "-y",
        "-i", audio_path,    
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        "-c", "copy",
        f"{TEMP_CHUNKS_DIR}/chunk_%03d.wav"
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(os.listdir(TEMP_CHUNKS_DIR))                          # RETURN THE NAMES OF THE CHUNK FILES IN ORDER (e.g., chunk_000.wav, chunk_001.wav, etc.)


def transcribe_chunk(chunk_path): # IT IS USED FOR TRANSCRIBING EACH CHUNK WITH RETRIES
    """Transcribe one chunk with retries"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=f,
                    model=MODEL
                )
            return result.text.strip()
        except Exception:
            print(f"      Retry {attempt}/{MAX_RETRIES}...")
            time.sleep(WAIT_SECONDS)

    return None


print("\n BATCH PROCESS STARTED")

for audio_file in os.listdir(AUDIO_DIR):   # HERE WE LOOP THROUGH ALL THE AUDIO FILES IN THE AUDIO DIRECTORY (WHICH WERE EXTRACTED FROM THE VIDEOS) AND PROCESS EACH ONE
    if not audio_file.endswith(".wav"):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file) # FULL PATH TO THE AUDIO FILE (E.G., audio/video1.wav)

    # ---- derive video id & title from filename ----
    video_id = audio_file.split("_")[0].replace(" ", "")
    title = audio_file.replace(".wav", "")

    print(f"\nProcessing video: {title}")

    # ---- split audio ----
    chunk_files = split_audio(audio_path)
    if not chunk_files:
        print("    Audio splitting failed, skipping video")
        continue

    print(f"   Created {len(chunk_files)} chunks")

    # ---- transcribe chunks ----
    all_chunks = []
    current_time = 0.0

    for idx, chunk in enumerate(chunk_files):
        chunk_path = os.path.join(TEMP_CHUNKS_DIR, chunk)
        print(f"   Transcribing chunk {idx+1}/{len(chunk_files)}")

        text = transcribe_chunk(chunk_path)
        if not text:
            print("      Chunk failed, skipping")
            current_time += CHUNK_SECONDS
            continue

        all_chunks.append({  # HERE WE ARE BUILDING A LIST OF CHUNKS WITH THEIR TRANSCRIPTIONS AND TIMESTAMPS, WHICH WILL LATER BE SAVED TO A JSON FILE
            "video_id": video_id,
            "title": title,
            "start": round(current_time, 2),
            "end": round(current_time + CHUNK_SECONDS, 2),
            "text": text
        })

        current_time += CHUNK_SECONDS

    if not all_chunks:
        print("    No chunks transcribed, skipping video")
        continue

    # ---- save JSON ----
    output = {
        "video_id": video_id,
        "title": title,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks
    }

    json_path = os.path.join(
        OUTPUT_JSON_DIR,
        f"{video_id}.json"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Saved JSON: {json_path}")

print("\n BATCH PROCESS COMPLETED")
