import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from .parser import fetch_html, extract_title_and_body, clean_text
from .features import compute_basic_metrics, compute_readability

def analyze_url(url, existing_features_df, tfidf_vectorizer, X_embeddings, threshold=0.5):
    """Analyze a given URL for quality and duplicates."""
    html = fetch_html(url)
    title, body = extract_title_and_body(html)
    body = clean_text(body)

    wc, sc = compute_basic_metrics(body)
    read = compute_readability(body)
    is_thin = wc < 500

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '../models/quality_model.pkl')
    model = load(model_path)

    # Predict content quality
    X_new = pd.DataFrame([{
        'word_count': wc,
        'sentence_count': sc,
        'flesch_reading_ease': read
    }])
    label = model.predict(X_new)[0]

    # Duplicate / similarity detection
    similar = []
    if tfidf_vectorizer is not None and X_embeddings is not None:
        X_new_vec = tfidf_vectorizer.transform([body])
        sims = cosine_similarity(X_new_vec, X_embeddings).flatten()
        top_idx = np.where(sims >= threshold)[0]
        for i in top_idx:
            similar.append({
                'url': existing_features_df.iloc[i]['url'],
                'similarity': float(sims[i])
            })

    return {
        'url': url,
        'title': title,
        'word_count': wc,
        'sentence_count': sc,
        'readability': read,
        'quality_label': label,
        'is_thin': is_thin,
        'similar_to': similar
    }
