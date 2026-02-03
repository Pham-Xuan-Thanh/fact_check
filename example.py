#!/usr/bin/env python3
"""
Example usage of the Fact-Checking Pipeline
"""
import asyncio
import os
from pipeline import FactCheckPipeline
from claims_detector import ClaimsDetector
from document_retriever import DocumentRetriever
from topk_selector import TopKSelector
from claim_verifier import ClaimVerifier
from config import GOOGLE_API_KEY, MODEL_NAME, SITE_CONFIGS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set API key
if GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

async def main():
    print("Initializing Fact-Checking Pipeline...")
    
    # Initialize LLM and embeddings
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize components
    claims_detector = ClaimsDetector(llm=llm)
    document_retriever = DocumentRetriever(embeddings=embeddings, site_configs=SITE_CONFIGS)
    topk_selector = TopKSelector(k=10)
    claim_verifier = ClaimVerifier(llm=llm)
    
    # Create pipeline
    pipeline = FactCheckPipeline(
        claims_detector=claims_detector,
        document_retriever=document_retriever,
        topk_selector=topk_selector,
        claim_verifier=claim_verifier
    )
    
    print("Pipeline initialized!\n")
    
    # Example texts to fact-check
    examples = [
        "Tổng thống Trung Quốc đến thăm Việt Nam",
        "The Earth is flat and NASA is hiding this fact",
        "Python is a programming language created by Guido van Rossum"
    ]
    
    for i, text in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {text}")
        print('='*60)
        
        try:
            result = await pipeline.run(text)
            
            print(f"\nFound {len(result.get('claims', []))} claim(s):\n")
            
            for claim in result.get('claims', []):
                print(f"Claim {claim['claim_id']}: {claim['claim_text']}")
                print(f"  Verdict: {claim.get('verdict', 'Unknown')}")
                print(f"  Confidence: {claim.get('confidence', 0):.2f}")
                
                evidence = claim.get('evidence', [])
                if evidence:
                    print(f"  Evidence sources: {len(evidence)}")
                    for doc in evidence[:2]:  # Show first 2 sources
                        print(f"    - {doc.get('source', 'Unknown')}: {doc.get('title', 'No title')[:60]}...")
                print()
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == '__main__':
    asyncio.run(main())
