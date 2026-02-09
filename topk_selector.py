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

    def select_top_k(self, claim: str, documents: List[Document], k: int) -> List[Document]:
        """
        Select top-k most relevant documents for a claim.
        Uses batched embedding calls to minimize API usage.

        Args:
            claim: The claim text
            documents: List of candidate documents
            k: Number of documents to select

        Returns:
            Top-k documents sorted by relevance
        """
        if not documents:
            return []

        if len(documents) <= k:
            return documents

        if self.embeddings is None:
            print("⚠️  No embeddings, returning first k documents")
            return documents[:k]

        try:
            # Single API call for claim
            claim_embedding = np.array(self.embeddings.embed_query(claim))
            # Single API call for ALL documents (batched)
            corpus = [doc.page_content for doc in documents]
            doc_embeddings = np.array(self.embeddings.embed_documents(corpus))

            # Normalize
            claim_norm = np.linalg.norm(claim_embedding)
            if claim_norm > 0:
                claim_embedding = claim_embedding / claim_norm

            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)
            doc_embeddings = doc_embeddings / doc_norms

            # Cosine similarity (vectorized, no loop)
            scores = np.dot(doc_embeddings, claim_embedding)
            top_indices = scores.argsort()[-k:][::-1]

            return [documents[idx] for idx in top_indices]

        except Exception as e:
            print(f"⚠️  Error in top-k selection: {str(e)[:50]}")
            return documents[:k]

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
