import os
import subprocess

VIDEO_DIR = "videos" # Directory containing the .mp4 files
AUDIO_DIR = "audio"  # Directory to save the extracted .wav files

os.makedirs(AUDIO_DIR, exist_ok=True) # Create the audio directory if it doesn't exist */

for file in os.listdir(VIDEO_DIR):    # Loop through all files in the video directory
    if file.endswith(".mp4"):          
        video_path = os.path.join(VIDEO_DIR, file)      # Full path to the video file
        audio_path = os.path.join(                      # Full path to the output audio file
            AUDIO_DIR,                                 # Audio directory
            file.replace(".mp4", ".wav")                  # Replace .mp4 extension with .wav for the output file name
        )

        print(f"Processing: {file}")

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",            # “FFmpeg, take this video, remove the video part, extract clean audio, convert it to ML-friendly format, and save it.”
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ])

print("All videos processed successfully")