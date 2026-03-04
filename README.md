## How to use this RAG AI Teaching assistant for your own data
## Step 1: Collect all your vides
 Move all your video files to videos folder

 ## Step 2: Convert to mp3
 convert all the files to mp3 by running video_to_mp3.py

 ## Step 3: Convert Mp3 to Json
 Convert all the mp3 files to json by running mp3_to_json.py

 ## Step 4: Convert all the json files to vector
 Use the preprocess_json.py to convert to json files to a data frame with embeddings and save it as embeddings.joblib

 ## Step 5: Prompt Generation and feeding to LLM: 
 Read the joblib file and load it into memory. Then Create a relevant prompt as per the user query and feed it to the LLM. 
