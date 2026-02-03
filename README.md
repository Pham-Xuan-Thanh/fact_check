# Fact-Checking Pipeline

A complete fact-checking system that extracts claims from text, searches for evidence, and verifies claims using LLMs.

## 🌟 Features

- **Claims Detection** - Extract verifiable claims from input text
- **Document Retrieval** - Search for relevant evidence from trusted sources
- **Top-K Selection** - Rank and select the most relevant documents
- **Claim Verification** - Verify claims as Supported/Refuted/Unknown
- **REST API** - Easy-to-use Flask API

## 📁 Project Structure

```
fact_check_repo/
├── claims_detector.py       # Extract claims from text
├── document_retriever.py    # Search and retrieve evidence
├── topk_selector.py         # Rank documents
├── claim_verifier.py        # Verify claims
├── pipeline.py              # Orchestrate the full pipeline
├── app.py                   # Flask API server
├── run.py                   # Main entry point
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup script
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will install:
- Python dependencies
- System libraries for Playwright
- Chromium browser

### 2. Configure

Set your Google API key:

```bash
export GOOGLE_API_KEY='your-google-api-key-here'
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run

```bash
python run.py
```

You'll see output like:

```
Starting Flask server...
Pipeline initialized successfully!
Flask server started on http://0.0.0.0:5000
```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "fact-check-pipeline",
  "version": "1.0.0"
}
```

### Fact-Check Text

```bash
curl -X POST http://localhost:5000/fact-check \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Tổng thống Trung Quốc đến thăm Việt Nam"
  }'
```

Response:
```json
{
  "claims": [
    {
      "claim_id": 1,
      "claim_text": "Tổng thống Trung Quốc đã đến thăm Việt Nam.",
      "verdict": "Supported",
      "confidence": 0.95,
      "evidence": [
        {
          "source": "tingia.gov.vn",
          "title": "...",
          "url": "..."
        }
      ]
    }
  ]
}
```

## 🐍 Python Usage

```python
import requests

API_URL = "http://localhost:5000"

# Fact-check some text
response = requests.post(
    f"{API_URL}/fact-check",
    json={"text": "Your text to fact-check"},
    headers={"Content-Type": "application/json"}
)

result = response.json()
print(result)
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **API Keys** - Google API key, SerpAPI key (optional)
- **Model** - LLM model name (default: gemini-2.5-flash-lite)
- **Server** - Host and port (default: 0.0.0.0:5000)
- **Retrieval** - Number of documents, sites to search
- **Site Configs** - Add more trusted sources

## 🔧 Advanced Usage

### Custom Pipeline

```python
from pipeline import FactCheckPipeline
from claims_detector import ClaimsDetector
from document_retriever import DocumentRetriever
from topk_selector import TopKSelector
from claim_verifier import ClaimVerifier
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import asyncio

# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

claims_detector = ClaimsDetector(llm=llm)
document_retriever = DocumentRetriever(embeddings=embeddings)
topk_selector = TopKSelector(k=10)
claim_verifier = ClaimVerifier(llm=llm)

pipeline = FactCheckPipeline(
    claims_detector,
    document_retriever,
    topk_selector,
    claim_verifier
)

# Run fact-checking
text = "Your text to fact-check"
result = asyncio.run(pipeline.run(text))
print(result)
```

## 📦 Dependencies

- **LangChain** - LLM orchestration
- **Google Generative AI** - Gemini models for claims detection & verification
- **Crawl4AI** - Web scraping
- **Playwright** - Browser automation
- **Flask** - API server

## 🐛 Troubleshooting

### Chromium fails to launch

Install missing dependencies:
```bash
sudo apt-get install -y libatk-bridge2.0-0 libatspi2.0-0
playwright install-deps chromium
```

### Port already in use

Change the port in `config.py`:
```python
PORT = 5001  # or any other available port
```

### API key not set

Set environment variable:
```bash
export GOOGLE_API_KEY='your-api-key'
```

Or edit `config.py`:
```python
GOOGLE_API_KEY = 'your-api-key-here'
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ for better fact-checking!
