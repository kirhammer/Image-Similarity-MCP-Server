# server.py

from fastapi import FastAPI
from pydantic import BaseModel
from find_similar_assets.image_similarity import find_similar_assets

app = FastAPI()

class FindSimilarRequest(BaseModel):
    imagePath: str
    assetsDirectory: str
    topK: int = 3

@app.post("/findSimilarAssets")
def find_similar(req: FindSimilarRequest):
    results = find_similar_assets(req.imagePath, req.assetsDirectory, req.topK)
    return {
        "matches": [
            {
                "file": match["name"],
                "path": match["path"],
                "similarity": match["similarity"]
            }
            for match in results
        ]
    }