from qdrant_client import QdrantClient
import streamlit as st

# CORRECT: Use the names of the keys you defined in secrets.toml
client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_KEY"]
)

# This will verify the connection and print your collections
try:
    print("Connecting to Qdrant...")
    collections = client.get_collections()
    print("Connection Successful!")
    print("Collections found:", collections)
except Exception as e:
    print(f"Connection failed: {e}")
