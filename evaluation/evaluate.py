import json
import re
import time
from pathlib import Path

from retrieval.rag import answer_question_with_sources


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation_results.json"

# Common words are excluded from the optional key-term coverage metric.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "does",
    "for", "from", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "what", "which", "with", "document", "answer", "mentioned",
}


def normalize_text(text):
    return re.findall(r"\b[a-z0-9]+(?:[.'-][a-z0-9]+)*\b", str(text or "").lower())


def calculate_answer_overlap(expected, actual):
    """Legacy lexical metric retained for comparison with earlier runs."""
    expected_words = set(normalize_text(expected))
    actual_words = set(normalize_text(actual))
    if not expected_words:
        return None
    return round(len(expected_words & actual_words) / len(expected_words), 3)


def extract_numbers(text):
    """Extract numeric values while normalizing commas and currency symbols."""
    values = re.findall(r"(?<![A-Za-z])\$?\s*\d[\d,]*(?:\.\d+)?%?", str(text or ""))
    normalized = set()
    for value in values:
        value = value.replace("$", "").replace(",", "").replace(" ", "")
        normalized.add(value)
    return sorted(normalized)


def extract_cited_pages(answer):
    pages = set()
    for pattern in (
        r"\bpages?\s*(?:number\s*)?[:#]?\s*(\d+)\b",
        r"\bp\.\s*(\d+)\b",
    ):
        pages.update(
            int(match.group(1))
            for match in re.finditer(pattern, str(answer or ""), re.IGNORECASE)
        )
    return sorted(pages)


def is_abstention(answer):
    """Heuristic only: a human should review ambiguous refusal cases."""
    text = " ".join(str(answer or "").lower().split())
    phrases = (
        "not mentioned in the document",
        "does not mention",
        "doesn't mention",
        "not specified in the document",
        "not provided in the document",
        "not available in the document",
        "not included in the document",
        "not included in the invoice",
        "cannot determine from the document",
        "can't determine from the document",
        "cannot answer based on the document",
        "i don't have enough information",
        "i do not have enough information",
        "not found in the document",
        "could not find sufficient information",
        "does not contain information",
        "does not contain the information",
        "evidence provided does not contain",
        "evidence does not contain",
        "i cannot provide that information",
        "unable to find",
        "insufficient information",
    )
    return any(phrase in text for phrase in phrases)


def key_term_coverage(expected, actual):
    """A second lexical signal that ignores common function words.

    This is not semantic or factual accuracy. It is useful alongside
    numeric checks and manual review, not as a standalone pass/fail score.
    """
    expected_terms = {
        token for token in normalize_text(expected)
        if token not in STOP_WORDS and not token.isdigit()
    }
    actual_terms = set(normalize_text(actual))
    if not expected_terms:
        return None
    return round(len(expected_terms & actual_terms) / len(expected_terms), 3)


def evaluate_test(test):
    question = test["question"]
    document_id = test["document_id"]
    expected_answer = test.get("expected_answer", "")
    expected_pages = test.get("expected_pages", [])
    should_abstain = len(expected_pages) == 0

    result = {
        "id": test.get("id", "unknown"),
        "question": question,
        "document_id": document_id,
        "expected_answer": expected_answer,
        "expected_pages": expected_pages,
    }

    try:
        start = time.perf_counter()
        response = answer_question_with_sources(question, document_id=document_id)
        latency = round(time.perf_counter() - start, 3)

        actual_answer = str(response.get("answer", ""))
        sources = response.get("sources", []) or []
        retrieved_pages = sorted({
            int(source["page_number"])
            for source in sources
            if source.get("page_number") is not None
        })
        cited_pages = extract_cited_pages(actual_answer)

        expected_numbers = extract_numbers(expected_answer)
        actual_numbers = extract_numbers(actual_answer)
        missing_numbers = [number for number in expected_numbers if number not in actual_numbers]

        overlap = calculate_answer_overlap(expected_answer, actual_answer)
        term_coverage = key_term_coverage(expected_answer, actual_answer)
        retrieval_hit = (
            any(page in retrieved_pages for page in expected_pages)
            if expected_pages else None
        )
        citation_hit = (
            any(page in cited_pages for page in expected_pages)
            if expected_pages else None
        )
        abstained = is_abstention(actual_answer)

        # Numeric check is only a signal: expected answers may contain dates,
        # page numbers, or other values that require case-specific interpretation.
        numeric_check = (len(missing_numbers) == 0) if expected_numbers else None

        result.update({
            "status": "completed",
            "actual_answer": actual_answer,
            "latency_seconds": latency,
            "answer_word_overlap": overlap,
            "key_term_coverage": term_coverage,
            "expected_numeric_values": expected_numbers,
            "actual_numeric_values": actual_numbers,
            "missing_expected_numeric_values": missing_numbers,
            "expected_numeric_values_present": numeric_check,
            "sources": sources,
            "retrieved_pages": retrieved_pages,
            "cited_pages": cited_pages,
            "page_retrieval_hit": retrieval_hit,
            "citation_page_hit": citation_hit,
            "should_abstain": should_abstain,
            "detected_abstention": abstained,
            "abstention_correct": (abstained == should_abstain),
        })
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})

    return result


def safe_average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 3) if values else None


def rate(hits, total):
    return round(hits / total, 3) if total else None


def calculate_summary(results):
    completed = [result for result in results if result.get("status") == "completed"]
    errors = [result for result in results if result.get("status") == "error"]
    retrieval = [r for r in completed if r.get("page_retrieval_hit") is not None]
    citations = [r for r in completed if r.get("citation_page_hit") is not None]
    abstention = [r for r in completed if r.get("should_abstain") is not None]
    numeric_checks = [r for r in completed if r.get("expected_numeric_values_present") is not None]

    retrieval_hits = sum(bool(r["page_retrieval_hit"]) for r in retrieval)
    citation_hits = sum(bool(r["citation_page_hit"]) for r in citations)
    abstention_hits = sum(bool(r["abstention_correct"]) for r in abstention)
    unanswerable = [r for r in completed if r.get("should_abstain") is True]
    answerable = [r for r in completed if r.get("should_abstain") is False]
    correct_refusals = sum(bool(r.get("detected_abstention")) for r in unanswerable)
    unexpected_refusals = sum(bool(r.get("detected_abstention")) for r in answerable)
    numeric_hits = sum(bool(r["expected_numeric_values_present"]) for r in numeric_checks)

    overlaps = [r["answer_word_overlap"] for r in completed if r.get("answer_word_overlap") is not None]
    term_coverages = [r["key_term_coverage"] for r in completed if r.get("key_term_coverage") is not None]
    latencies = [r["latency_seconds"] for r in completed if r.get("latency_seconds") is not None]

    return {
        "total_tests": len(results),
        "completed_tests": len(completed),
        "error_tests": len(errors),
        "average_latency_seconds": safe_average(latencies),
        "average_answer_word_overlap_legacy": safe_average(overlaps),
        "average_key_term_coverage": safe_average(term_coverages),
        "page_retrieval_hits": retrieval_hits,
        "page_retrieval_tests": len(retrieval),
        "page_retrieval_rate": rate(retrieval_hits, len(retrieval)),
        "citation_page_hits": citation_hits,
        "citation_page_tests": len(citations),
        "citation_page_rate": rate(citation_hits, len(citations)),
        "abstention_classification_correct": abstention_hits,
        "abstention_classification_tests": len(abstention),
        "abstention_classification_accuracy": rate(abstention_hits, len(abstention)),
        "unanswerable_tests": len(unanswerable),
        "correct_refusals_on_unanswerable": correct_refusals,
        "answerable_tests": len(answerable),
        "unexpected_refusals_on_answerable": unexpected_refusals,
        "numeric_value_hits": numeric_hits,
        "numeric_value_tests": len(numeric_checks),
        "numeric_value_rate": rate(numeric_hits, len(numeric_checks)),
    }


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        print("Error: test_dataset.json must contain a JSON list.")
        return

    results = []
    print("\nStarting DocuMind AI evaluation using answer evidence and expanded metrics...\n")

    for index, test in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] Running {test.get('id', 'unknown')}...")
        result = evaluate_test(test)
        results.append(result)
        print(f"Status: {result['status']}")

        if result["status"] == "completed":
            print(f"Latency: {result['latency_seconds']} seconds")
            print(f"Evidence pages: {result['retrieved_pages']}")
            print(f"Cited pages: {result['cited_pages']}")
            print(f"Legacy word overlap: {result['answer_word_overlap']}")
            print(f"Key-term coverage: {result['key_term_coverage']}")
            print(f"Expected numeric values present: {result['expected_numeric_values_present']}")
            print(f"Should abstain: {result.get('should_abstain')}")
            print(f"Detected abstention: {result.get('detected_abstention')}")
            print(f"Abstention classification correct: {result.get('abstention_correct')}")
        else:
            print(f"Error: {result.get('error')}")
        print("-" * 50)

    summary = calculate_summary(results)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "results": results}, file, indent=4, ensure_ascii=False)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Completed: {summary['completed_tests']}")
    print(f"Errors: {summary['error_tests']}")
    print(f"Average latency: {summary['average_latency_seconds']} seconds")
    print(f"Legacy word overlap: {summary['average_answer_word_overlap_legacy']}")
    print(f"Average key-term coverage: {summary['average_key_term_coverage']}")
    print(f"Page retrieval: {summary['page_retrieval_hits']}/{summary['page_retrieval_tests']}")
    print(f"Citation page correctness: {summary['citation_page_hits']}/{summary['citation_page_tests']}")
    print(f"Abstention classification accuracy: {summary['abstention_classification_correct']}/{summary['abstention_classification_tests']}")
    print(f"Correct refusals on unanswerable tests: {summary['correct_refusals_on_unanswerable']}/{summary['unanswerable_tests']}")
    print(f"Unexpected refusals on answerable tests: {summary['unexpected_refusals_on_answerable']}/{summary['answerable_tests']}")
    print(f"Expected numeric values present: {summary['numeric_value_hits']}/{summary['numeric_value_tests']}")
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
