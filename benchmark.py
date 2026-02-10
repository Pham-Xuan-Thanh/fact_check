#!/usr/bin/env python3
"""
Benchmark the Fact-Checking Pipeline on the ViFactCheck dataset.

Data source: data/sampled_data.csv
Uses:
  - Statement column as the input claim
  - Url column as the reference document for retrieval
  - labels column as ground truth (0=REFUTED, 1=SUPPORTED, 2=NEI)
"""

import asyncio
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from claim_verifier import ClaimVerifier
from config import EMBEDDING_MODEL, GOOGLE_API_KEY, MODEL_NAME
from select_evidence import process_evidence
from topk_selector import TopKSelector

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
# Dataset: 0 = REFUTED, 1 = SUPPORTED, 2 = NOT ENOUGH INFO
LABEL_ID_TO_NAME = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_ID_TO_NAME.items()}
LABEL_NAMES = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("benchmark_results")

# Retry config for rate-limited API calls
MAX_RETRIES = 5
BASE_RETRY_DELAY = 60  # seconds


# ===================================================================
# Retry helper for rate-limited LLM / embedding calls
# ===================================================================
def _extract_retry_delay(error_msg: str) -> float | None:
    """Parse 'retryDelay' or 'Please retry in Xs' from the error message."""
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


async def call_with_retry(func, *args, **kwargs):
    """Call *func* and retry on 429 RESOURCE_EXHAUSTED with backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                delay = _extract_retry_delay(err) or BASE_RETRY_DELAY * attempt
                delay = min(delay + 5, 300)  # add buffer, cap at 5 min
                print(f"    [Rate limited] attempt {attempt}/{MAX_RETRIES}, "
                      f"waiting {delay:.0f}s ...")
                await asyncio.sleep(delay)
            else:
                raise
    # final attempt – let it raise
    return func(*args, **kwargs)


# ===================================================================
# Document retrieval from a specific URL
# ===================================================================
async def fetch_document_from_url(url: str, timeout: int = 30) -> str | None:
    """Fetch the main text content from a URL using crawl4ai, with
    requests+BeautifulSoup as fallback."""

    # --- Attempt 1: crawl4ai ---
    try:
        async with AsyncWebCrawler(
            browser_type="chromium", headless=True, verbose=False
        ) as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, "markdown") and result.markdown:
                text = result.markdown.strip()
                if len(text) > 100:
                    return text[:5000]
            if result and hasattr(result, "html") and result.html:
                soup = BeautifulSoup(result.html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    return text[:5000]
    except Exception:
        pass

    # --- Attempt 2: requests + BeautifulSoup ---
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        # Try common article selectors first
        for selector in [
            "article",
            ".content",
            ".detail",
            ".post-content",
            "#content",
            "main",
        ]:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    return text[:5000]
        # Fallback: full body text
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 100:
            return text[:5000]
    except Exception:
        pass

    return None


# ===================================================================
# Process a single sample through the pipeline
# ===================================================================
async def process_sample(
    idx: int,
    statement: str,
    url: str,
    context: str,
    ground_truth: int,
    topk_selector: TopKSelector,
    claim_verifier: ClaimVerifier,
    evidence_k: int = 2,
    evidence_window_size: int = 2,
) -> Dict[str, Any]:
    """Run one sample through the pipeline (skip claims detection).

    Uses both URL scraping and the Context column from the dataset.
    URL content is tried first; Context is used as fallback if scraping fails.
    When both are available, they are combined for richer evidence.
    """

    result: Dict[str, Any] = {
        "index": idx,
        "statement": statement,
        "url": url,
        "ground_truth_id": ground_truth,
        "ground_truth_label": LABEL_ID_TO_NAME.get(ground_truth, "UNKNOWN"),
        "predicted_label": None,
        "predicted_id": None,
        "confidence": None,
        "reasoning": None,
        "error": None,
        "content_source": None,
    }

    # --- Step 1: Build document content from URL + Context ---
    url_content = await fetch_document_from_url(url)
    context_content = context.strip() if isinstance(context, str) and context.strip() else None

    if url_content and context_content:
        # Combine both sources for richer evidence
        content = url_content + "\n\n" + context_content
        result["content_source"] = "url+context"
    elif url_content:
        content = url_content
        result["content_source"] = "url"
    elif context_content:
        content = context_content
        result["content_source"] = "context"
    else:
        result["error"] = "No content from URL or Context"
        result["predicted_label"] = "NOT ENOUGH INFO"
        result["predicted_id"] = 2
        result["confidence"] = 0.0
        return result

    domain = urlparse(url).netloc
    doc = Document(
        page_content=content,
        metadata={"source": domain, "url": url, "title": domain, "date": None},
    )

    # --- Step 2: Top-k selection (single doc, so just passes through) ---
    top_docs = topk_selector.select_top_k(statement, [doc], k=3)

    # --- Step 3: Evidence selection ---
    doc_list = []
    for d in top_docs:
        doc_list.append(
            {
                "score": 1.0,
                "site": d.metadata.get("source", "unknown"),
                "title": d.metadata.get("title", "Untitled"),
                "content": d.page_content,
                "url": d.metadata.get("url", ""),
                "date": d.metadata.get("date", ""),
            }
        )

    data_for_evidence = [{"claim": statement, "documents": doc_list}]
    selected_evidences = process_evidence(
        data_for_evidence, k=evidence_k, window_size=evidence_window_size
    )

    if not selected_evidences or not selected_evidences[0].get("evidences"):
        result["error"] = "No evidence selected"
        result["predicted_label"] = "NOT ENOUGH INFO"
        result["predicted_id"] = 2
        result["confidence"] = 0.0
        return result

    # --- Step 4: Claim verification ---
    evidences_for_verifier = []
    for i, ev in enumerate(selected_evidences[0]["evidences"]):
        evidences_for_verifier.append(
            {
                "evidence_id": f"evidence_{i}",
                "site": ev.get("url", "Unknown"),
                "text": ev.get("content", ""),
                "date": ev.get("date", ""),
                "reason": f"Selected evidence {i+1}",
            }
        )

    verification = await call_with_retry(
        claim_verifier.verify_claim, statement, evidences_for_verifier
    )

    predicted_label = verification.get("label", "NOT ENOUGH INFO")
    result["predicted_label"] = predicted_label
    result["predicted_id"] = LABEL_NAME_TO_ID.get(predicted_label, 2)
    result["confidence"] = verification.get("confidence", 0.0)
    result["reasoning"] = verification.get("reasoning", "")

    # Propagate error from claim_verifier (it swallows exceptions internally)
    if verification.get("error"):
        result["error"] = verification["error"]
        print(f"  [LLM ERROR] sample {idx}: {verification['error']}")

    return result


# ===================================================================
# Compute and print metrics
# ===================================================================
def compute_metrics(results: List[Dict[str, Any]], results_dir: Path) -> Dict[str, Any]:
    """Compute classification metrics and save a report."""

    y_true = [r["ground_truth_id"] for r in results if r["predicted_id"] is not None]
    y_pred = [r["predicted_id"] for r in results if r["predicted_id"] is not None]

    if not y_true:
        print("No valid predictions to evaluate.")
        return {}

    accuracy = accuracy_score(y_true, y_pred)

    # Per-class and averaged metrics
    report_dict = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, labels=[0, 1, 2], output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, labels=[0, 1, 2], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    f1_macro = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2], zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)

    metrics = {
        "total_samples": len(results),
        "evaluated_samples": len(y_true),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
    }

    # --- Count errors ---
    error_count = sum(1 for r in results if r.get("error"))
    metrics["error_count"] = error_count

    # --- Print report ---
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nTotal samples:     {metrics['total_samples']}")
    print(f"Evaluated samples: {metrics['evaluated_samples']}")
    print(f"Errors:            {metrics['error_count']}")
    print(f"\nAccuracy:          {metrics['accuracy']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"\nClassification Report:\n{report_str}")
    print(f"Confusion Matrix (rows=true, cols=pred):")
    print(f"Labels: {LABEL_NAMES}")
    for row in cm:
        print(f"  {row}")
    print("=" * 70)

    # --- Save metrics ---
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {metrics_path}")

    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"ViFactCheck Benchmark Report\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Total samples:     {metrics['total_samples']}\n")
        f.write(f"Evaluated samples: {metrics['evaluated_samples']}\n")
        f.write(f"Errors:            {metrics['error_count']}\n\n")
        f.write(f"Accuracy:          {metrics['accuracy']:.4f}\n")
        f.write(f"F1 (macro):        {metrics['f1_macro']:.4f}\n")
        f.write(f"F1 (weighted):     {metrics['f1_weighted']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):    {metrics['recall_macro']:.4f}\n\n")
        f.write(f"Classification Report:\n{report_str}\n\n")
        f.write(f"Confusion Matrix (rows=true, cols=pred):\n")
        f.write(f"Labels: {LABEL_NAMES}\n")
        for row in cm:
            f.write(f"  {row.tolist()}\n")
    print(f"Report saved to {report_path}")

    return metrics


# ===================================================================
# Main benchmark runner
# ===================================================================
async def run_benchmark(
    max_samples: int | None = None,
    delay_between_samples: float = 1.0,
):
    """
    Run the benchmark on the ViFactCheck test set.

    Args:
        max_samples: Limit number of samples (None = all). Useful for testing.
        delay_between_samples: Seconds to wait between samples (API rate limit).
    """

    # --- Setup output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    print(f"Results will be saved to: {results_dir}")

    # --- Load dataset ---
    data_path = Path("data/sampled_data.csv")
    print(f"\nLoading dataset from {data_path}...")
    dataset = pd.read_csv(data_path)
    print(f"Loaded {len(dataset)} samples")

    if max_samples:
        dataset = dataset.head(max_samples)
        print(f"Using first {len(dataset)} samples")

    # --- Initialize components ---
    print("\nInitializing pipeline components...")
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    topk_selector = TopKSelector(embeddings=embeddings)
    claim_verifier = ClaimVerifier(llm=llm)
    print("Pipeline components initialized.")

    # --- Check for existing progress (resume support) ---
    progress_file = results_dir / "progress.json"
    all_results: List[Dict[str, Any]] = []
    completed_indices: set = set()

    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        completed_indices = {r["index"] for r in all_results}
        print(f"Resuming: {len(completed_indices)} samples already completed")

    # --- Process samples ---
    total = len(dataset)
    start_time = time.time()

    for i in range(total):
        if i in completed_indices:
            continue

        row = dataset.iloc[i]
        statement = str(row.get("Statement", ""))
        url = str(row.get("Url", ""))
        context = str(row.get("Context", ""))
        label = int(row.get("labels", -1))

        elapsed = time.time() - start_time
        done = len(completed_indices)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0

        print(
            f"\n[{done + 1}/{total}] "
            f"(elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m) "
            f"Processing: {statement[:80]}..."
        )

        try:
            result = await process_sample(
                idx=i,
                statement=statement,
                url=url,
                context=context,
                ground_truth=label,
                topk_selector=topk_selector,
                claim_verifier=claim_verifier,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            result = {
                "index": i,
                "statement": statement,
                "url": url,
                "ground_truth_id": label,
                "ground_truth_label": LABEL_ID_TO_NAME.get(label, "UNKNOWN"),
                "predicted_label": "NOT ENOUGH INFO",
                "predicted_id": 2,
                "confidence": 0.0,
                "reasoning": None,
                "error": str(e),
            }

        all_results.append(result)
        completed_indices.add(i)

        # Log prediction
        gt = result["ground_truth_label"]
        pred = result["predicted_label"]
        match = "OK" if result["ground_truth_id"] == result["predicted_id"] else "MISS"
        print(f"  => GT: {gt} | Pred: {pred} | {match}")

        # Save individual prediction
        pred_path = predictions_dir / f"sample_{i:04d}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save progress periodically (every 10 samples)
        if len(completed_indices) % 10 == 0:
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  [Progress saved: {len(completed_indices)}/{total}]")

        # Rate limiting
        if delay_between_samples > 0:
            await asyncio.sleep(delay_between_samples)

    # --- Save final results ---
    final_results_path = results_dir / "all_results.json"
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {final_results_path}")

    # --- Compute metrics ---
    metrics = compute_metrics(all_results, results_dir)

    # --- Save config used ---
    config_path = results_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "max_samples": max_samples,
                "delay_between_samples": delay_between_samples,
                "total_samples_processed": len(all_results),
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )

    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time/60:.1f} minutes")
    print(f"Results directory: {results_dir}")

    return metrics


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark fact-checking pipeline on ViFactCheck")
    parser.add_argument(
        "-n", "--max-samples", type=int, default=None,
        help="Max number of samples to process (default: all)",
    )
    parser.add_argument(
        "-d", "--delay", type=float, default=1.0,
        help="Delay between samples in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(max_samples=args.max_samples, delay_between_samples=args.delay))
