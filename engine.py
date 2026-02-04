import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
from qdrant_client import QdrantClient, models

class GMRS_Engine:
    def __init__(self, qdrant_url, qdrant_key, collection_name):
        self.collection_name = collection_name

        # Connect to Qdrant
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)

        # Load Local Model (Free, runs on CPU/GPU)
        print("📥 Loading local CLIP model (this may take a minute first time)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('clip-ViT-B-32', device=device)
        print(f"✅ Model loaded on {device}")

    def ensure_collection(self):
        """Creates a collection with Two Named Vectors (Text & Image)"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text_vec": models.VectorParams(size=512, distance=models.Distance.COSINE),
                    "image_vec": models.VectorParams(size=512, distance=models.Distance.COSINE),
                }
            )
            print(f"✅ Collection '{self.collection_name}' created.")

    def get_text_embedding(self, text):
        return self.model.encode(text).tolist()

    def get_image_embedding(self, image_url):
        try:
            # Download image to memory for processing
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return self.model.encode(img).tolist()
        except Exception as e:
            print(f"⚠️ Image embed error ({image_url}): {e}")
            raise

    def hybrid_search(self, text_query, image_url, limit=4):
        """
        Hybrid Search using Local Embeddings + Qdrant
        """
        # 1. Generate Embeddings Locally
        text_vec = self.get_text_embedding(text_query)
        image_vec = self.get_image_embedding(image_url) if image_url else None

        # 2. Parallel Search (Text)
        text_results = self.client.query_points(
            collection_name=self.collection_name,
            using="text_vec",
            query=text_vec,
            limit=limit * 2,
            with_payload=True
        ).points

        # 3. Parallel Search (Image - if available)
        image_results = []
        if image_vec:
            image_results = self.client.query_points(
                collection_name=self.collection_name,
                using="image_vec",
                query=image_vec,
                limit=limit * 2,
                with_payload=True
            ).points

        # 4. RRF Fusion
        k = 60
        scores = {}
        point_map = {}

        for rank, point in enumerate(text_results):
            point_map[point.id] = point
            scores[point.id] = scores.get(point.id, 0) + (1 / (k + rank + 1))

        for rank, point in enumerate(image_results):
            point_map[point.id] = point
            scores[point.id] = scores.get(point.id, 0) + (1 / (k + rank + 1))

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        return [point_map[pid] for pid in sorted_ids]
