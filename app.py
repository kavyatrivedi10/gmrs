import streamlit as st
from engine import GMRS_Engine
from feedback_loop import generative_feedback_loop

# --- SETUP ---
st.set_page_config(page_title="Unified AI Search", layout="wide")

try:
    engine = GMRS_Engine(
        st.secrets["QDRANT_URL"],
        st.secrets["QDRANT_KEY"]
    )
except KeyError as e:
    st.error("Missing secrets. Please check .streamlit/secrets.toml")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Controls")
    enable_gfl = st.toggle("Enable GFL (Generative Loop)", value=True)
    st.caption("When enabled, the AI visualizes your intent before searching.")

# --- MAIN UI ---
st.title("🛍️ Unified Multimodal Search")
st.markdown("Query → Intent Refinement → Image Generation → Hybrid Search")

if query := st.chat_input("Ex: A futuristic sneaker with neon lights"):

    # CONTAINER FOR THE WORKFLOW
    with st.status("Processing Workflow...", expanded=True) as status:

        # 1. INTENT REFINEMENT & IMAGE GENERATION
        if enable_gfl:
            status.write("🧠 Step 1: Gemini is refining your intent...")
            refined_text, ai_image_url = generative_feedback_loop(query)
            status.write(f"🎨 Step 2: Image generated based on: *{refined_text}*")
        else:
            refined_text = query
            ai_image_url = None
            status.write("⏩ GFL Skipped. Using raw text.")

        # 2. HYBRID SEARCH (Text + Image)
        status.write("🔍 Step 3: Running Hybrid Search (Text + Image Fusion)...")
        results = engine.hybrid_search(
            text_query=refined_text,
            image_url=ai_image_url,
            limit=4
        )
        status.update(label="Workflow Complete!", state="complete", expanded=False)

    # --- DISPLAY RESULTS ---

    # A. Refined Intent
    st.subheader("1. Refined Intent")
    st.info(refined_text)

    # B. Generated Image
    if ai_image_url:
        st.subheader("2. AI Visualization")
        st.image(ai_image_url, width=300, caption="The AI is looking for something like this.")

    # C. Search Results
    st.subheader("3. Top Matches (Hybrid)" if ai_image_url else "2. Top Matches (Text Only)")
    if results:
        cols = st.columns(4)
        for i, point in enumerate(results):
            prod = point.payload
            with cols[i]:
                # Display Product Image
                st.image(prod.get('image', ''), width='stretch')
                # Display Product Name
                st.markdown(f"**{prod.get('product_name', 'Unknown')}**")
                # Display Price
                st.caption(f"Price: {prod.get('price', 'N/A')}")
                # Link
                with st.expander("Link"):
                    st.write(prod.get('url', '#'))
    else:
        st.warning("No matches found.")
