import re
import numpy as np
import textstat

def compute_basic_metrics(text):
    """Compute word count and sentence count."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    return len(words), len(sentences)

def compute_readability(text):
    """Compute Flesch Reading Ease safely."""
    if not isinstance(text, str) or len(text.split()) < 50:
        return 0
    try:
        score = textstat.flesch_reading_ease(text)
        if np.isnan(score):
            return 0
        return round(score, 2)
    except Exception:
        return 0
