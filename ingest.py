import pandas as pd
import streamlit as st
import ast
from qdrant_client import models
from engine import GMRS_Engine

# --- CONFIG ---
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_KEY = st.secrets["QDRANT_KEY"]

def clean_image_url(image_str):
    try:
        if pd.isna(image_str):
            return None
        urls = ast.literal_eval(image_str)
        if isinstance(urls, list) and len(urls) > 0:
            return urls[0]
        return None
    except:
        return None

def main():
    """
    Ingests CSV into Qdrant Collection.
    This may be a lengthy process.
    Thus, also maintains a progress log in files `embedded.txt` & `skipped.txt`.
    """
    print("🚀 Starting Local Ingestion...")
    # Initialize Local Engine
    engine = GMRS_Engine(QDRANT_URL, QDRANT_KEY)
    engine.ensure_collection()

    try:
        df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    except FileNotFoundError:
        print("❌ CSV not found.")
        return

    # Check progress till now
    try:
        embedded = engine.client.get_collection(engine.collection_name).points_count
    except:
        embedded = 0
    skipped = sum(1 for _ in open("skipped.txt", "rb"))
    current_count = embedded + skipped
    print(f"ℹ️ Resuming from index {current_count}...")
    with open("embedded.txt", "a") as embedded_log, open("skipped.txt", "a") as skip_log:
        # Loop
        for idx, row in df.iloc[current_count:].iterrows():
            image_url = clean_image_url(row['image'])
            desc = f"{row['product_name']} {row['description']}"[:500] # Truncate to avoid model overflow

            # Collect embeddings into a dict as per Qdrant requirements
            vector_dict = {}
            try:
                image_vec = engine.get_image_embedding(image_url) if image_url else None
                vector_dict["image_vec"] = image_vec
            except Exception as e:
                print(f"⚠️ Embedding error at index {idx}: {e}")
                skip_log.write(f"{idx}\n")
                continue
            text_vec = engine.get_text_embedding(desc)
            vector_dict["text_vec"] = text_vec

            try:
                # Upsert into the database
                engine.client.upsert(
                    collection_name=engine.collection_name,
                    points=[
                        models.PointStruct(
                            id=idx,
                            vector=vector_dict,
                            payload={
                                "product_name": row['product_name'],
                                "price": row['discounted_price'],
                                "image": image_url,
                                "url": row['product_url']
                            }
                        )
                    ]
                )
            except Exception as e:
                print(f"❌ Upsert error at index {idx}: {e}")
                skip_log.write(f"{idx}\n")
                continue

            # Print progress for the user to see
            if idx % 10 == 0:
                print(f"✅ Indexed {idx}")
            embedded_log.write(f"{idx}\n")

if __name__ == "__main__":
    main()
