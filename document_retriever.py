import os
import re
from typing import Any, List, Dict
from urllib.parse import quote_plus
import numpy as np
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

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
            "date": ".date, time",
        },
    },
]


class DocumentRetriever:
    """
    Retrieves relevant documents for each claim using web search and scraping.
    """

    def __init__(
        self, embeddings, knowledge_base=None, site_configs=None, serpapi_key=None
    ):
        """
        Args:
            embeddings: LangChain embedding model for semantic search
            knowledge_base: Optional pre-built vector store or document collection
            site_configs: List of site configurations for web scraping
            serpapi_key: Optional SerpAPI key for Google search fallback
        """
        self.embeddings = embeddings
        self.knowledge_base = knowledge_base
        self.vectorstore = None
        self.site_configs = site_configs or SITE_CONFIGS
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.claim_documents = {}  # Cache documents per claim
        self.claim_embeddings = {}  # Cache embeddings per claim

    def build_vectorstore(self, documents: List[Document]):
        """
        Build a vector store from documents.

        Args:
            documents: List of LangChain Document objects
        """
        if documents:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print(f"Built vector store with {len(documents)} documents")

    def _build_search_query(self, claim: str) -> str:
        """Extract keywords from claim for search."""
        claim_lower = claim.lower()
        stop_words = {
            "là",
            "của",
            "và",
            "có",
            "này",
            "được",
            "đã",
            "trong",
            "cho",
            "về",
            "với",
            "một",
            "các",
            "trên",
            "từ",
            "đến",
            "sau",
            "bị",
            "cũng",
            "như",
            "khi",
            "vào",
            "ra",
            "để",
            "theo",
            "sẽ",
            "bởi",
            "tại",
        }
        words = re.findall(r"\b\w+\b", claim_lower)
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return " ".join(keywords[:5])

    def _extract_article_links(
        self, html: str, base_domain: str, config: Dict, max_links: int = 30
    ) -> List[str]:
        """Extract article URLs using site-specific selectors."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            urls = []
            seen = set()

            if "search_results_selector" in config and "result_link_selector" in config:
                result_items = soup.select(config["search_results_selector"])
                for item in result_items:
                    try:
                        link = item.select_one(config["result_link_selector"])
                        if link and link.get("href"):
                            href = link["href"]
                            if href.startswith("/"):
                                href = f"https://{base_domain}{href}"
                            elif href.startswith("//"):
                                href = f"https:{href}"
                            if (
                                href.startswith("http")
                                and base_domain in href
                                and href not in seen
                            ):
                                seen.add(href)
                                urls.append(href)
                                if len(urls) >= max_links:
                                    break
                    except Exception:
                        continue
            return urls
        except Exception as e:
            print(f"  ⚠️  Error extracting links: {str(e)[:50]}")
            return []

    def _search_google(
        self, query: str, domain: str, max_results: int = 10
    ) -> List[str]:
        """Fallback: Search Google for articles from specific domain using SerpAPI."""
        if not self.serpapi_key:
            return []
        try:
            params = {
                "engine": "google",
                "q": f"{query} site:{domain}",
                "api_key": self.serpapi_key,
                "num": max_results,
                "gl": "vn",
                "hl": "vi",
            }
            response = requests.get(
                "https://serpapi.com/search", params=params, timeout=10
            )
            data = response.json()
            urls = []
            for result in data.get("organic_results", [])[:max_results]:
                url = result.get("link")
                if url and domain in url:
                    urls.append(url)
            return urls
        except Exception as e:
            print(f"  ⚠️  Google search failed: {str(e)[:50]}")
            return []

    def _extract_content(self, html: str, selectors: Dict) -> Dict:
        """Extract article content using CSS selectors."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            result = {}
            for field, selector in selectors.items():
                try:
                    for sel in selector.split(","):
                        elem = soup.select_one(sel.strip())
                        if elem:
                            result[field] = elem.get_text(separator="\n", strip=True)
                            break
                except Exception:
                    continue
            return result
        except Exception as e:
            print(f"  ⚠️  Error extracting content: {str(e)[:50]}")
            return {}

    async def _crawl_articles(self, claim: str, max_per_site: int = 5) -> List[Dict]:
        """Search and scrape articles."""
        query = self._build_search_query(claim)
        print(f"🔍 Searching for: '{query}'")
        documents = []

        async with AsyncWebCrawler(
            browser_type="chromium", headless=True, verbose=False
        ) as crawler:
            for config in self.site_configs:
                print(f"📰 Searching {config['site']}...")
                try:
                    search_url = config["search_url"].format(query=quote_plus(query))
                    search_result = await crawler.arun(url=search_url)
                    article_urls = self._extract_article_links(
                        search_result.html,
                        config["site"],
                        config,
                        max_links=max_per_site * 2,
                    )
                    print(f"  Found {len(article_urls)} result links")

                    if len(article_urls) == 0 and self.serpapi_key:
                        print("  🔄 Trying Google search fallback...")
                        article_urls = self._search_google(
                            query, config["site"], max_results=max_per_site * 2
                        )
                        print(f"  Found {len(article_urls)} links from Google")

                    count = 0
                    for url in article_urls:
                        if count >= max_per_site:
                            break
                        try:
                            result = await crawler.arun(url=url)
                            if not result or not hasattr(result, "html"):
                                continue

                            content = self._extract_content(
                                result.html, config["selectors"]
                            )

                            # Fallback to markdown if content extraction failed
                            if not content.get("content"):
                                markdown = getattr(result, "markdown", None)
                                if markdown:
                                    content["content"] = markdown[:3000]
                                    content["title"] = (
                                        markdown.split("\n")[0][:80]
                                        if markdown
                                        else "Untitled"
                                    )

                            # Only add if we have actual content
                            if (
                                content.get("content")
                                and len(content["content"].strip()) > 50
                            ):
                                documents.append(
                                    {
                                        "site": config["site"],
                                        "url": url,
                                        "title": content.get("title") or "Untitled",
                                        "content": content["content"],
                                        "date": content.get("date"),
                                    }
                                )
                                count += 1
                                print(f"  ✓ {content.get('title', 'Article')[:60]}")
                        except Exception as e:
                            print(f"  ⚠️  Error crawling {url[:50]}: {str(e)[:30]}")
                            continue
                    print(f"  → {count} articles\n")
                except Exception as e:
                    print(f"  ✗ {str(e)[:60]}\n")

        print(f"✅ Retrieved {len(documents)} total documents\n")
        return documents

    async def retrieve_documents(self, claim: str, k: int = 10) -> List[Document]:
        """
        Retrieve relevant documents for a claim.

        Args:
            claim: The claim text to search for
            k: Number of documents to retrieve

        Returns:
            List of LangChain Document objects
        """
        # Crawl articles
        raw_docs = await self._crawl_articles(claim, max_per_site=5)

        if not raw_docs:
            print("⚠️  No documents retrieved")
            return []

        # Convert to LangChain Documents
        langchain_docs = [
            Document(
                page_content=f"{doc['title']}\n\n{doc['content']}",
                metadata={
                    "source": doc.get("site", "unknown"),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", "Untitled"),
                    "date": doc.get("date"),
                },
            )
            for doc in raw_docs
            if doc.get("content")
        ]

        if not langchain_docs:
            print("⚠️  No valid documents after conversion")
            return []

        # Build embeddings for ranking
        try:
            corpus = [doc.page_content for doc in langchain_docs]
            doc_embeddings = self.embeddings.embed_documents(corpus)
            claim_embedding = self.embeddings.embed_query(claim)
        except Exception as e:
            print(f"⚠️  Error generating embeddings: {str(e)[:50]}")
            return langchain_docs[:k]  # Return without ranking

        try:
            # Convert to numpy arrays and normalize
            doc_embeddings = np.array(doc_embeddings)
            claim_embedding = np.array(claim_embedding)

            # Normalize embeddings
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)  # Avoid division by zero
            doc_embeddings = doc_embeddings / doc_norms

            claim_norm = np.linalg.norm(claim_embedding)
            if claim_norm > 0:
                claim_embedding = claim_embedding / claim_norm

            # Calculate similarity scores
            scores = np.dot(doc_embeddings, claim_embedding)
            top_indices = scores.argsort()[-k:][::-1]

            # Return top-k documents
            return [langchain_docs[idx] for idx in top_indices]
        except Exception as e:
            print(f"⚠️  Error calculating similarity: {str(e)[:50]}")
            return langchain_docs[:k]

    async def retrieve_for_all_claims(
        self, claims: List[Dict[str, Any]], k: int = 10
    ) -> Dict[int, List[Document]]:
        """
        Retrieve documents for all claims.

        Args:
            claims: List of claim dictionaries
            k: Number of documents per claim

        Returns:
            Dictionary mapping claim_id to list of documents
        """
        results = {}

        for claim_data in claims:
            claim_id = claim_data["claim_id"]
            claim_text = claim_data["claim_text"]

            print(f"\n{'=' * 70}")
            print(f"Retrieving for Claim {claim_id}: {claim_text}")
            print(f"{'=' * 70}\n")

            documents = await self.retrieve_documents(claim_text, k=k)
            results[claim_id] = documents

            print(f"Retrieved {len(documents)} documents for claim {claim_id}\n")

        return results
