from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document


class TopKSelector:
    """
    Selects top-k most relevant documents for each claim.
    """

    def __init__(self, llm=None, embeddings=None):
        """
        Args:
            llm: Optional LLM for reranking
            embeddings: LangChain embeddings for similarity scoring
        """
        self.llm = llm
        self.embeddings = embeddings

        if self.embeddings is None:
            print("⚠️  Warning: No embeddings provided to TopKSelector. Relevance scoring will not work.")

    def calculate_relevance_score(self, claim: str, document: Document) -> float:
        """
        Calculate relevance score between claim and document.

        Args:
            claim: The claim text
            document: A document

        Returns:
            Relevance score
        """
        if self.embeddings is None:
            raise ValueError("Embeddings model is required for relevance scoring. Please provide embeddings when initializing TopKSelector.")

        # Use cosine similarity with embeddings
        claim_embedding = self.embeddings.embed_query(claim)
        doc_embedding = self.embeddings.embed_documents([document.page_content])[0]

        # Normalize and calculate cosine similarity
        claim_embedding = np.array(claim_embedding)
        doc_embedding = np.array(doc_embedding)
        claim_embedding = claim_embedding / np.linalg.norm(claim_embedding)
        doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)

        score = np.dot(claim_embedding, doc_embedding)
        return float(score)

    def select_top_k(self, claim: str, documents: List[Document], k: int) -> List[Document]:
        """
        Select top-k most relevant documents for a claim.

        Args:
            claim: The claim text
            documents: List of candidate documents
            k: Number of documents to select

        Returns:
            Top-k documents sorted by relevance
        """
        if not documents:
            return []

        # Score all documents
        scored_docs = []
        for doc in documents:
            score = self.calculate_relevance_score(claim, doc)
            scored_docs.append((score, doc))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top-k documents
        return [doc for score, doc in scored_docs[:k]]

    def select_for_all_claims(self,
                              claims: List[Dict[str, Any]],
                              claim_documents: Dict[int, List[Document]],
                              k: int) -> Dict[int, List[Document]]:
        """
        Select top-k documents for all claims.

        Args:
            claims: List of claims
            claim_documents: Documents retrieved for each claim
            k: Number of documents to keep per claim

        Returns:
            Dictionary mapping claim_id to top-k documents
        """
        results = {}

        for claim_data in claims:
            claim_id = claim_data['claim_id']
            claim_text = claim_data['claim_text']
            documents = claim_documents.get(claim_id, [])

            print(f"Selecting top-{k} documents for claim {claim_id}...")
            top_docs = self.select_top_k(claim_text, documents, k)
            results[claim_id] = top_docs
            print(f"Selected {len(top_docs)} documents\n")

        return results