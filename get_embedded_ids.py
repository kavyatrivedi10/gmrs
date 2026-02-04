import streamlit as st
from engine import GMRS_Engine

def get_embedded_ids(engine=None):
    """
    Generator function to get the IDs of all successfully embedded points
    """
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_KEY = st.secrets["QDRANT_KEY"]
    COLLECTION_NAME = "flipkart_local_clip"

    if engine is None:
        engine = GMRS_Engine(QDRANT_URL, QDRANT_KEY, COLLECTION_NAME)

    points = engine.client.scroll(collection_name=engine.collection_name,
        with_payload=False,
        with_vectors=False,
        limit=20000
    )[0]
    return (int(point.id) for point in points)

if __name__ == "__main__":
    print('\n'.join(str(id) for id in get_embedded_ids()))
