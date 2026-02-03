# 🚀 Quick Start Guide

## Installation

1. **Extract the zip file:**
```bash
unzip fact_check_repo.zip
cd fact_check_repo
```

2. **Run setup (installs all dependencies):**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Set your Google API Key:**
```bash
export GOOGLE_API_KEY='your-api-key-here'
```
Get your API key from: https://makersuite.google.com/app/apikey

4. **Run the server:**
```bash
python run.py
```

Server will be available at: `http://localhost:5000`

## Test It

```bash
# Health check
curl http://localhost:5000/health

# Fact-check
curl -X POST http://localhost:5000/fact-check \
  -H 'Content-Type: application/json' \
  -d '{"text": "Python is a programming language"}'
```

## Files Structure

```
fact_check_repo/
├── 📄 claims_detector.py       # Extracts claims from text
├── 📄 document_retriever.py    # Searches for evidence
├── 📄 topk_selector.py         # Ranks documents
├── 📄 claim_verifier.py        # Verifies claims
├── 📄 pipeline.py              # Orchestrates everything
├── 🌐 app.py                   # Flask API server
├── ▶️  run.py                   # Main entry point
├── ⚙️  config.py                # Configuration
├── 📋 requirements.txt         # Python packages
├── 🛠️  setup.sh                 # Setup script
├── 📖 README.md                # Full documentation
├── 📝 example.py               # Example usage
└── 🚫 .gitignore               # Git ignore rules
```

## Usage Options

### Option 1: Run Server
```bash
python run.py
```
Runs on http://localhost:5000.

### Option 2: Python Code
```python
from pipeline import FactCheckPipeline
# ... (see example.py)
```

## Troubleshooting

**Problem:** Chromium fails to launch  
**Solution:**
```bash
sudo apt-get install -y libatk-bridge2.0-0
playwright install-deps chromium
```

**Problem:** Port 5000 in use  
**Solution:** Edit `config.py` and change `PORT = 5001`

**Problem:** API key error  
**Solution:** 
```bash
export GOOGLE_API_KEY='your-key'
# or edit config.py directly
```

## Next Steps

- Read `README.md` for full documentation
- See `example.py` for Python usage
- Customize `config.py` for your needs
- Add more sites in `SITE_CONFIGS`

---

Need help? Check the README or open an issue!
