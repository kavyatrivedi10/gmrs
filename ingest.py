import pandas as pd
import streamlit as st
import ast
from qdrant_client import models
from engine import GMRS_Engine
from get_embedded_ids import get_embedded_ids
import os

# --- CONFIG ---
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_KEY = st.secrets["QDRANT_KEY"]
COLLECTION_NAME = "flipkart_local_clip"

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
    Thus, also maintains a log of failures in file `skipped.txt`.
    """
    print("🚀 Starting Local Ingestion...")
    # Initialize Local Engine
    engine = GMRS_Engine(QDRANT_URL, QDRANT_KEY, COLLECTION_NAME)
    engine.ensure_collection()

    try:
        df = pd.read_csv("flipkart_com-ecommerce_sample.csv")
    except FileNotFoundError:
        print("❌ CSV not found.")
        return

    last_embedded: int
    try:
        last_embedded = max(get_embedded_ids(engine), default=-1)
    except Exception as e:
        print(f"⚠️ Failed to retrieve embedded ids.\n\t{e}")
        last_embedded = -1

    if os.path.exists("skipped.txt"):
        last_skipped = max((int(x) for x in open("skipped.txt", "rb+")), default=-1)
    else:
        last_skipped = -1
    last_touched = max(last_embedded, last_skipped)
    print(f"ℹ️ Resuming from index {last_touched + 1}...")

    with open("skipped.txt", "a") as log:
        # Loop
        for idx, row in df.iloc[last_touched + 1:].iterrows():
            image_url = clean_image_url(row['image'])
            desc = f"{row['product_name']} {row['description']}"[:500] # Truncate to avoid model overflow

            # Collect embeddings into a dict as per Qdrant requirements
            vector_dict = {}
            try:
                image_vec = engine.get_image_embedding(image_url) if image_url else None
                vector_dict["image_vec"] = image_vec
            except Exception as e:
                print(f"⚠️ Embedding error at index {idx}.\n\t{e}")
                log.write(f"{idx}\n")
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
                log.write(f"{idx}\n")
                continue

            # Print progress for the user to see
            if idx % 20 == 0:
                print(f"✅ Indexed {idx}")
            log.write(f"{idx}\n")

if __name__ == "__main__":
    main()
