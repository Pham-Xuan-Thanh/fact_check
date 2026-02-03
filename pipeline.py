import asyncio
from typing import Dict, Any


class FactCheckPipeline:
    """
    Complete fact-checking pipeline.
    """

    def __init__(self,
                 claims_detector: ClaimsDetector,
                 document_retriever: DocumentRetriever,
                 top_k_selector: TopKSelector,
                 claim_verifier: ClaimVerifier,
                 top_k: int = 5):
        self.claims_detector = claims_detector
        self.document_retriever = document_retriever
        self.top_k_selector = top_k_selector
        self.claim_verifier = claim_verifier
        self.top_k = top_k

    async def run_async(self, text: str) -> Dict[str, Any]:
        """
        Run the complete fact-checking pipeline (async version).

        Args:
            text: Input text to fact-check

        Returns:
            Complete pipeline results
        """
        results = {}

        # Step 1: Detect claims
        print("=" * 70)
        print("Step 1: Detecting claims...")
        print("=" * 70)
        claims = self.claims_detector.detect_claims(text)
        results['claims'] = claims
        print(f"✅ Found {len(claims)} claims\n")

        if not claims:
            print("⚠️  No claims detected. Exiting pipeline.")
            return results

        # Step 2: Retrieve documents (ASYNC)
        print("=" * 70)
        print("Step 2: Retrieving documents...")
        print("=" * 70)
        claim_documents = await self.document_retriever.retrieve_for_all_claims(
            claims, k=self.top_k * 2  # Retrieve more for better selection
        )
        results['retrieved_documents'] = claim_documents
        print(f"✅ Retrieved documents for all claims\n")

        # Step 3: Select top-k documents
        print("=" * 70)
        print(f"Step 3: Selecting top-{self.top_k} documents...")
        print("=" * 70)
        top_k_documents = self.top_k_selector.select_for_all_claims(
            claims, claim_documents, self.top_k
        )
        results['top_k_documents'] = top_k_documents
        print(f"✅ Selected top-{self.top_k} documents for all claims\n")

        # Step 4: Verify claims
        print("=" * 70)
        print("Step 4: Verifying claims...")
        print("=" * 70)

        # Convert documents to evidence format for verifier
        claims_with_evidence = []
        for claim_data in claims:
            claim_id = claim_data['claim_id']
            claim_text = claim_data['claim_text']
            documents = top_k_documents.get(claim_id, [])

            # Convert LangChain Documents to evidence format
            evidences = []
            for idx, doc in enumerate(documents):
                evidences.append({
                    "evidence_id": f"doc_{idx}_{doc.metadata.get('source', 'unknown')}",
                    "site": doc.metadata.get('url', 'Unknown'),
                    "text": doc.page_content[:500],  # Limit to 500 chars
                    "reason": f"Retrieved as relevant document {idx+1}"
                })

            claims_with_evidence.append({
                "claim": claim_text,
                "evidences": evidences
            })

        verification_results = self.claim_verifier.verify_all_claims(claims_with_evidence)
        results['verification_results'] = verification_results
        print(f"✅ Verified all claims\n")

        return results

    def run(self, text: str) -> Dict[str, Any]:
        """
        Run the complete fact-checking pipeline (sync wrapper).

        Args:
            text: Input text to fact-check

        Returns:
            Complete pipeline results
        """
        # Check if running in an existing event loop (like in Jupyter/Colab)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.run_async(text))
        else:
            # Event loop already running (Jupyter/Colab), use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.run_async(text))

    def display_results(self, results: Dict[str, Any]):
        """
        Display pipeline results in a readable format.

        Args:
            results: Pipeline results dictionary
        """
        print("\n" + "=" * 80)
        print("FACT-CHECK RESULTS")
        print("=" * 80)

        claims = results.get('claims', [])
        verification_results = results.get('verification_results', [])

        for i, (claim, verification) in enumerate(zip(claims, verification_results)):
            print(f"\n{'─' * 80}")
            print(f"CLAIM {i+1}: {claim['claim_text']}")
            print(f"{'─' * 80}")

            label = verification.get('label', 'UNKNOWN')
            confidence = verification.get('confidence', 0.0)
            reasoning = verification.get('reasoning', 'No reasoning provided')

            # Color coding for labels
            label_color = {
                'SUPPORTED': '✅',
                'REFUTED': '❌',
                'NOT ENOUGH INFO': '⚠️'
            }

            print(f"{label_color.get(label, '❓')} Label: {label}")
            print(f"🎯 Confidence: {confidence:.2%}")
            print(f"💭 Reasoning: {reasoning}")

            # Display evidences
            evidences = verification.get('evidences', [])
            if evidences:
                print(f"\n📚 Evidence ({len(evidences)} sources):")
                for j, evidence in enumerate(evidences[:3], 1):  # Show top 3
                    print(f"\n  [{j}] {evidence.get('evidence_id', 'Unknown')}")
                    print(f"      🔗 {evidence.get('site', 'Unknown')[:60]}...")
                    print(f"      📄 {evidence.get('text', '')[:100]}...")

        print("\n" + "=" * 80)

        # Summary
        if verification_results:
            supported = sum(1 for v in verification_results if v.get('label') == 'SUPPORTED')
            refuted = sum(1 for v in verification_results if v.get('label') == 'REFUTED')
            not_enough = sum(1 for v in verification_results if v.get('label') == 'NOT ENOUGH INFO')

            print(f"\n📊 SUMMARY:")
            print(f"   ✅ Supported: {supported}")
            print(f"   ❌ Refuted: {refuted}")
            print(f"   ⚠️  Not Enough Info: {not_enough}")
            print(f"   📝 Total Claims: {len(claims)}")

        print("\n" + "=" * 80 + "\n")