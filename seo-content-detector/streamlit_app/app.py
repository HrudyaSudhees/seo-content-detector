import streamlit as st
import pandas as pd
import os
import sys
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.scorer import analyze_url

# ---------------------------------------------------
# Path setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# ---------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------
st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="wide")
st.title("🔍 SEO Content Quality & Duplicate Detector")
st.write("Analyze webpage content for SEO quality and detect duplicates.")

# ---------------------------------------------------
# Load Data & Model
# ---------------------------------------------------
try:
    model = load(os.path.join(BASE_DIR, "models", "quality_model.pkl"))
    st.sidebar.success("✅ Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model: {e}")
    st.stop()

try:
    features = pd.read_csv(os.path.join(PARENT_DIR, "data", "features.csv"))
    st.sidebar.info(f"📄 Dataset loaded: {len(features)} pages")
except Exception as e:
    st.sidebar.error(f"❌ Could not load dataset: {e}")
    st.stop()

# ---------------------------------------------------
# Load or Build TF-IDF Vectorizer
# ---------------------------------------------------
try:
    tfidf = load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
    st.sidebar.success("✅ TF-IDF Vectorizer loaded.")
except Exception:
    st.sidebar.warning("⚠️ TF-IDF not found, building from scratch...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf.fit(features['body_text'].fillna(''))

X_embeddings = tfidf.transform(features['body_text'].fillna(''))

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
url = st.text_input("Enter a webpage URL to analyze:")
threshold = st.slider("Duplicate detection threshold (cosine similarity)", 0.3, 0.9, 0.5, 0.05)

if st.button("Analyze"):
    with st.spinner("Analyzing content..."):
        result = analyze_url(
            url,
            existing_features_df=features,
            tfidf_vectorizer=tfidf,
            X_embeddings=X_embeddings,
            threshold=threshold
        )
    st.subheader("🔎 Analysis Result")
    st.json(result)

    st.metric("Predicted Quality Label", result['quality_label'])
    st.metric("Word Count", result['word_count'])
    st.metric("Readability Score", result['readability'])

    if result['similar_to']:
        st.subheader("📑 Similar / Duplicate Pages")
        st.dataframe(pd.DataFrame(result['similar_to']))
    else:
        st.info("No similar pages found above the threshold.")
