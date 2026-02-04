import streamlit as st
import urllib.parse
import requests

def generative_feedback_loop(user_query):
    """
    1. Refines user intent using Google Gemini via Pollinations.ai
    2. Generates a product visualization using ChatGPT via Pollinations.ai
    """
    api_key = st.secrets.get("POLLINATIONS_API_KEY", "")

    # Textual Refinement Prompt
    refine_prompt = (
        f"User search: '{user_query}'. "
        "Task: Rewrite this into a precise, professional e-commerce product description to remove ambiguity. "
        "Make as few changes as possible to the user's original intent. "
        "Do NOT add any color on your own but keep the user's choices. "
        "Keep the description AS GENERIC AS POSSIBLE while still retaining the user's intent, and professional language. "
        "Describe exactly ONE product, not an array of products. "
        "Keep it concise -- under 30 words. "
        "Do not use flowery language. "
        "Output ONLY the description. "
        "Keep the description simple and reader-friendly."
    )

    # Free text generation using Gemini via Pollinations.ai
    try:
        text_url = "https://gen.pollinations.ai/v1/chat/completions"
        response = requests.post(
            text_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gemini-fast",
                "messages": [{"role": "user", "content": refine_prompt}],
                "temperature": 0.7
            },
            timeout=10
        )
        response.raise_for_status()
        refined_intent = response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        refined_intent = user_query # Fall back to original input if error

    # Free visual generation via Pollinations.ai
    # The image is generated based on the Gemini-refined description
    encoded_prompt = urllib.parse.quote("The image is professional with studio lighting. " + refined_intent)
    base_url = "https://gen.pollinations.ai/image"
    params = f"?model=gptimage&nologo=true&key={api_key}"
    generated_image_url = f"{base_url}/{encoded_prompt}{params}"

    return refined_intent, generated_image_url

