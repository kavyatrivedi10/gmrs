import streamlit as st
from engine import GMRS_Engine

# --- CONFIG ---
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_KEY = st.secrets["QDRANT_KEY"]
COLLECTION_NAME = "flipkart_local_clip"

if __name__ == "__main__":
    """
    Print the total number of points stored in the collection.
    """
    engine = GMRS_Engine(QDRANT_URL, QDRANT_KEY, COLLECTION_NAME)
    print(engine.client.get_collection(engine.collection_name).points_count)

