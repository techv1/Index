import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import numpy as np
import faiss

app = FastAPI()

# Pulled securely from OpenShift Environment Settings
REPO_ID = os.getenv("HF_REPO_ID", "Vbvinc/Kuchbhi_dataset")
HF_TOKEN = os.getenv("HF_TOKEN")

index = None
video_ids = None

print("Downloading files from Dataset Hub...")
try:
    index_path = hf_hub_download(repo_id=REPO_ID, filename="faiss.index", repo_type="dataset", token=HF_TOKEN)
    index = faiss.read_index(index_path)
    print(f"SUCCESS: FAISS Index Loaded. Expected dimensions: {index.d}")

    video_ids_path = hf_hub_download(repo_id=REPO_ID, filename="video_ids.npy", repo_type="dataset", token=HF_TOKEN)
    video_ids = np.load(video_ids_path, allow_pickle=True)
    print("All files loaded successfully!")
except Exception as e:
    print(f"Error loading data files: {str(e)}")

class SearchQuery(BaseModel):
    vector: list
    k: int = 5

@app.post("/search")
def search(query: SearchQuery):
    global index, video_ids
    if index is None or video_ids is None:
        raise HTTPException(status_code=500, detail="Database files are not loaded yet.")
    try:
        query_vector = np.array([query.vector], dtype=np.float32)
        if query_vector.shape[1] != index.d:
            raise HTTPException(status_code=400, detail=f"Dimension mismatch! Expected {index.d}")
            
        distances, indices = index.search(query_vector, query.k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                video_id = video_ids[idx].tolist() if hasattr(video_ids[idx], 'tolist') else video_ids[idx]
                results.append({"matching_index": int(idx), "distance": float(dist), "video_id": video_id})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "FAISS API Active on OpenShift", "dimensions": index.d if index else "Loading"}
