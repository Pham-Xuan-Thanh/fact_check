"""
Fact-Checking API Server with Flask
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os

# Import pipeline components
from pipeline import FactCheckPipeline
from claims_detector import ClaimsDetector
from document_retriever import DocumentRetriever
from topk_selector import TopKSelector
from claim_verifier import ClaimVerifier
from config import PORT, HOST, GOOGLE_API_KEY, MODEL_NAME, SITE_CONFIGS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set API key
if GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Initialize components
print("Initializing Fact-Checking Pipeline...")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

claims_detector = ClaimsDetector(llm=llm)
document_retriever = DocumentRetriever(embeddings=embeddings, site_configs=SITE_CONFIGS)
topk_selector = TopKSelector(k=10)
claim_verifier = ClaimVerifier(llm=llm)

pipeline = FactCheckPipeline(
    claims_detector=claims_detector,
    document_retriever=document_retriever,
    topk_selector=topk_selector,
    claim_verifier=claim_verifier
)

print("Pipeline initialized successfully!")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'fact-check-pipeline',
        'version': '1.0.0'
    })


@app.route('/fact-check', methods=['POST'])
def fact_check():
    """Main fact-checking endpoint"""
    try:
        # Get input text
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        input_text = data['text']
        
        # Run pipeline
        import asyncio
        result = asyncio.run(pipeline.run(input_text))
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server():
    """Run Flask server"""
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == '__main__':
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"Flask server started on http://{HOST}:{PORT}")
    
    # Keep main thread alive
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down server...")
