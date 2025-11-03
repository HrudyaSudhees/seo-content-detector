import re
from bs4 import BeautifulSoup
import requests

def fetch_html(url):
    """Fetch raw HTML from a URL."""
    headers = {'User-Agent': 'seo-content-detector/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.text if response.status_code == 200 else ''
    except Exception:
        return ''

def extract_title_and_body(html):
    """Extract title and body text from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else ''
    body = soup.get_text(separator=' ')
    return title, body

def clean_text(text):
    """Clean HTML text for analysis."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()
