import json
import os
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


def _extract_json_array(raw: str) -> List[Any]:
    if raw is None:
        raise ValueError("Empty LLM response")
    raw = raw.strip()

    if raw.startswith("[") and raw.endswith("]"):
        return json.loads(raw)

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in response")

    return json.loads(raw[start : end + 1])


class ClaimsDetector:
    """
    Detects and extracts verifiable claims from input text.
    Each claim includes a single keyword for retrieval (Step 2).
    """

    def __init__(self, llm=None, model_name: str = None):
        if llm is None:
            model = model_name or os.getenv("MODEL_NAME") or "gemini-2.5-flash-lite"
            llm = ChatGoogleGenerativeAI(model=model)

        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Split the following paragraph into a list of atomic, self-contained factual claims.
            Each claim must express exactly one verifiable fact.
            IMPORTANT: The output language must match the input language (do NOT translate).
            For each claim, provide:
            - exactly one short keyword/phrase for web search
            - a list of subjects/entities mentioned in the claim (person/organization/place/thing)
            - one primary domain/topic the claim is about

            Text: {text}

            Return ONLY a JSON list with the following format:
            [
                {{"claim_id": 1, "claim_text": "...", "keyword": "...", "subject": ["..."], "domain": "..."}},
                {{"claim_id": 2, "claim_text": "...", "keyword": "...", "subject": ["..."], "domain": "..."}}
            ]
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def detect_claims(self, text: str) -> List[Dict[str, Any]]:
        raw = self.chain.run(text=text)
        items = _extract_json_array(raw)

        if not isinstance(items, list):
            raise ValueError("Claims response is not a list")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            claim_text = item.get("claim_text")
            keyword = item.get("keyword")
            subject = item.get("subject")
            domain = item.get("domain")
            claim_id = item.get("claim_id") or idx

            if not isinstance(claim_text, str) or not claim_text.strip():
                continue
            if not isinstance(keyword, str) or not keyword.strip():
                continue
            if not isinstance(subject, list) or not subject:
                continue
            subjects = [s.strip() for s in subject if isinstance(s, str) and s.strip()]
            if not subjects:
                continue
            if not isinstance(domain, str) or not domain.strip():
                continue

            normalized.append(
                {
                    "claim_id": int(claim_id),
                    "claim_text": claim_text.strip(),
                    "keyword": keyword.strip(),
                    "subject": subjects,
                    "domain": domain.strip(),
                }
            )

        if not normalized:
            raise ValueError("No valid claims parsed from response")

        return normalized

