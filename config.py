"""
Configuration settings for the Fact-Checking Pipeline
"""
import os

# API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')  # Set your API key here or via environment variable
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/gemini-embedding-001')

# Server Configuration
HOST = '0.0.0.0'
PORT = 8501

# Retrieval Configuration
MAX_DOCUMENTS_PER_SITE = 3
TOP_K_DOCUMENTS = 5

# Site configurations with search URLs and specific selectors
SITE_CONFIGS = [
    {
        "site": "tingia.gov.vn",
        "search_url": "https://tingia.gov.vn/tim-kiem?key={query}",
        "search_results_selector": ".tc-post-list-style3 .item",  # Each search result item
        "result_link_selector": ".title a",  # Link within each item
        "selectors": {
            "title": "h1, .title",
            "content": ".content, article, .detail",
            "date": ".date, time"
        }
    },
]

# Optional: SerpAPI key for Google search fallback
SERPAPI_KEY = os.getenv('SERPAPI_KEY', "")
