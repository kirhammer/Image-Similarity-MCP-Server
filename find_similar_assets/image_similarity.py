# image_similarity.py

from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn.functional import cosine_similarity
import os
import pickle

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'embeddings_cache.pkl')

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

def get_embedding(image_path):
    cache = load_cache()
    image_path_str = str(image_path)
    mtime = os.path.getmtime(image_path_str)
    cache_entry = cache.get(image_path_str)
    if cache_entry and cache_entry['mtime'] == mtime:
        return torch.tensor(cache_entry['embedding'])
    image = Image.open(image_path)
    # Handle palette images with transparency
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    normed = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    cache[image_path_str] = {
        'embedding': normed.cpu().numpy(),
        'mtime': mtime
    }
    save_cache(cache)
    return normed

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