# image_similarity.py

from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn.functional import cosine_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_embedding(image_path):
    image = Image.open(image_path)
    # Handle palette images with transparency
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

def find_similar_assets(image_path: str, assets_dir: str, top_k: int = 3):
    query_embedding = get_embedding(image_path)
    asset_embeddings = []

    for img_path in Path(assets_dir).rglob("*.[pj][np]g"):  # .png/.jpg
        try:
            emb = get_embedding(img_path)
            sim = cosine_similarity(query_embedding, emb).item()
            asset_embeddings.append({
                "name": img_path.name,
                "path": str(img_path),
                "similarity": sim
            })
        except Exception:
            continue  # ignore corrupt or unreadable files

    return sorted(asset_embeddings, key=lambda x: x["similarity"], reverse=True)[:top_k]