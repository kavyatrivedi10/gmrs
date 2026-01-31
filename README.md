# E-Commerce Recommender System

## A novel approach based on a Generative Feedback Loop

## Tech Stack

1. Interface & Processing: Gemini 2.5 Flash and GPTImage via Pollinations.ai
2. Dataset: Found an existing dataset picked up from Kaggle at https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
3. Vector Database: Qdrant to store vector embedding of the products data of the Kaggle dataset
4. Backend: Native Python
5. Frontend: Streamlit

### **Team Members:**
- Kavya Trivedi
- Poornashree H V

### Problem statement

- Shift from traditional search to conversational and customized product discovery.
- Enhance user experience and confidence from conventional search.
- Need for personalized recommendations.

### Solution

- Generative AI-powered product search.
- Refines user query into a more descriptive text through Gemini 2.5 Flash.
- Generates an image based on the descriptive text.
- Search for similar products using both text and image embeddings (multimodal search).
- Repeat until the user's satisfaction.

# Setup

### Env File

Create a .streamlit/secrets.toml file at the root directory which should have the following variables-

`QDRANT_API_KEY = "YOUR_API_KEY_HERE"
`

`QDRANT_URL = "DATABASE_URL_HERE"
`

`POLLINATIONS_API_KEY = "YOUR_API_KEY_HERE"
`
For this, you will need to sign up on Qdrant and Pollinations.ai.

### Backend Dependencies

Setup a virtual python enviroment.
`python -m venv venv`

Activate the enviroment
`source venv/bin/activate`

Install backend dependancies using
`pip install -r requirements.txt`

### Database

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) into the root directory.
Ensure that it is named as `flipkart_com-ecommerce_sample.csv` (this is the default download name).
Run `python ingest.py` or `python3 ingest.py` to ingest the data into the Qdrant database.

### Frontend

Set the frontend running using `streamlit run app`.
