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
            You are a fact-checking assistant.

            Task: Extract the MOST IMPORTANT factual claims from the paragraph, not a sentence-by-sentence split.

            Definitions:
            - "Atomic claim" = one checkable proposition. It MAY include closely related details (numbers, dates, locations, sources, comparisons) only when those details are necessary to keep the claim meaningful and self-contained.
            - "Important claim" = central to the paragraph’s message OR high-impact OR likely controversial OR includes specific quantitative/temporal info OR names notable entities OR implies causation/blame.

            Rules:
            1) Output language MUST match input language exactly. Do NOT translate.
            2) DO NOT mechanically split by sentences.
            3) MERGE related facts into ONE claim when they refer to the same event/trend/study finding/announcement/relationship.
            - Combine result + supporting details (date/number/source) if they describe the same fact.
            - If two sentences are dependent to make sense, combine them.
            4) DROP low-value details (setup, background, vague opinions, filler, rhetorical questions) unless essential for verifying the main claim.
            5) Output AT MOST 5 claims. If more than 5 candidates exist, select the 5 most important/check-worthy.
            6) Each claim must be self-contained and understandable without the original paragraph.
            7) Avoid vague claims. Prefer specific, verifiable statements. If the text is too vague, output fewer claims (0–2 is acceptable).

            For each claim, output:
            - claim_id: id of claim
            - claim_text: 1–2 sentences max, combined if needed
            - keyword: EXACTLY ONE short keyword/phrase suitable for web search (NOT a full sentence)
            - subject: list of subjects/entities mentioned (person/organization/place/thing). Use [] if none.
            - domain: EXACTLY ONE primary topic/domain (e.g., politics, public health, economics, technology, climate, crime, sports, business, science)

            Input: {text}

            Return ONLY a JSON array in this exact schema:
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

