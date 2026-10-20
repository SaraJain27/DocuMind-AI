
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.chunking import chunk_pages


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("data/indexes")

# Load the embedding model once.
model = SentenceTransformer(MODEL_NAME)


def get_index_paths(document_id: str = "default"):
    """Return separate index and metadata paths for each document."""

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", document_id)

    if not safe_id:
        raise ValueError("Invalid document ID.")

    index_path = INDEX_DIR / f"{safe_id}.faiss"
    metadata_path = INDEX_DIR / f"{safe_id}.json"

    return index_path, metadata_path


def build_index(
    pages: list[dict],
    document_id: str = "default",
) -> int:
    """Create and save a separate FAISS index for a document."""

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text was found to index.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path, metadata_path = get_index_paths(document_id)

    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    return len(chunks)


def search_documents(
    query: str,
    top_k: int = 5,
    document_id: str = "default",
    min_similarity: float = 0.25,
) -> list[dict]:
    """Retrieve relevant chunks from one document's FAISS index."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1.")

    index_path, metadata_path = get_index_paths(document_id)

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No index found for document '{document_id}'. "
            "Build its index first."
        )

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal == 0:
        return []

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_vector = np.asarray(query_vector, dtype=np.float32)

    candidate_k = min(max(top_k * 10, 20), index.ntotal)

    scores, indices = index.search(query_vector, candidate_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        similarity = float(score)

        if similarity < min_similarity:
            continue

        results.append({
            **chunks[int(idx)],
            "similarity": similarity,
        })

    
    # Keep keyword matches for debugging only; ranking uses cosine similarity.
    query_terms = {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", query)
        if len(word) > 2
    }

    for item in results:
        text_lower = item["text"].lower()

        keyword_matches = sum(
            1
            for term in query_terms
            if re.search(rf"\b{re.escape(term)}\b", text_lower)
        )

        item["keyword_matches"] = keyword_matches
        item["combined_score"] = item["similarity"]

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]

def compare_document_pages(
    document_ids: list[str],
    top_k: int = 10,
    min_similarity: float = 0.55,
) -> list[dict]:
    """Find semantically similar pages across selected documents."""

    if len(set(document_ids)) < 2:
        raise ValueError("Select at least two different documents.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1 and 1.")

    document_pages = {}

    # Read the existing metadata for each selected document.
    for document_id in document_ids:
        index_path, metadata_path = get_index_paths(document_id)

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No index found for document '{document_id}'."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        # Combine chunks belonging to the same page.
        pages = {}

        for chunk in chunks:
            page_number = chunk["page_number"]
            pages.setdefault(page_number, []).append(chunk["text"])

        document_pages[document_id] = {
            page_number: "\n".join(texts)
            for page_number, texts in pages.items()
        }

    # Create one record per page.
    page_records = []

    for document_id, pages in document_pages.items():
        for page_number, text in pages.items():
            if text.strip():
                page_records.append({
                    "document_id": document_id,
                    "page_number": page_number,
                    "text": text,
                })

    if len(page_records) < 2:
        return []

    # Embed all pages in a batch.
    embeddings = model.encode(
        [page["text"] for page in page_records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    matches = []

    # Compare pages from different documents only.
    for i, page_a in enumerate(page_records):
        for j in range(i + 1, len(page_records)):
            page_b = page_records[j]

            if page_a["document_id"] == page_b["document_id"]:
                continue

            similarity = float(np.dot(embeddings[i], embeddings[j]))

            if similarity >= min_similarity:
                matches.append({
                    "document_a": page_a["document_id"],
                    "page_a": page_a["page_number"],
                    "text_a": page_a["text"][:500],
                    "document_b": page_b["document_id"],
                    "page_b": page_b["page_number"],
                    "text_b": page_b["text"][:500],
                    "similarity": similarity,
                })

    matches.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return matches[:top_k]





def find_exact_page_matches(
    query: str,
    document_id: str,
) -> list[dict]:
    """
    Find pages matching a topic, prioritizing exact phrases
    and headings over pages containing scattered query words.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    _, metadata_path = get_index_paths(document_id)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata found for document '{document_id}'."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    stop_words = {
        "what", "which", "where", "when", "who", "why",
        "is", "are", "was", "were", "the", "a", "an",
        "on", "in", "at", "of", "for", "to", "does",
        "do", "did", "page", "pages", "find", "show",
        "tell", "me", "please", "have", "has", "had",
        "same", "discussed", "explained", "mentioned",
        "covered", "appears", "located", "about",
    }

    query_words = re.findall(r"[a-z0-9]+", query.lower())
    topic_words = [
        word for word in query_words
        if word not in stop_words
    ]

    if not topic_words:
        topic_words = query_words

    topic_phrase = " ".join(topic_words)

    # Combine chunks belonging to the same page.
    pages = {}

    for chunk in chunks:
        page_number = chunk["page_number"]
        pages.setdefault(page_number, []).append(chunk["text"])

    matches = []

    for page_number, texts in pages.items():
        page_text = "\n".join(texts)

        # Normalize extracted PDF text.
        normalized_text = " ".join(
            re.findall(r"[a-z0-9]+", page_text.lower())
        )

        lines = [
            " ".join(re.findall(r"[a-z0-9]+", line.lower()))
            for line in page_text.splitlines()
        ]
        lines = [line for line in lines if line]

        # Highest priority: topic appears as a complete heading line.
        heading_match = any(
            line == topic_phrase
            for line in lines
        )

        # Next priority: the complete topic phrase occurs in page text.
        phrase_match = bool(topic_phrase) and (
            f" {topic_phrase} " in f" {normalized_text} "
        )

        # Lower priority: every topic word occurs somewhere on the page.
        matched_words = [
            word for word in topic_words
            if re.search(
                rf"\b{re.escape(word)}\b",
                normalized_text,
            )
        ]

        all_words_match = (
            bool(topic_words)
            and len(matched_words) == len(topic_words)
        )

        if heading_match:
            match_type = "exact heading"
            priority = 3
        elif phrase_match:
            match_type = "exact phrase"
            priority = 2
        elif all_words_match:
            match_type = "all topic words"
            priority = 1
        else:
            continue

        matches.append({
            "document_id": document_id,
            "page_number": page_number,
            "match_type": match_type,
            "priority": priority,
            "text": page_text[:500],
        })

    # Show the strongest matches first.
    matches.sort(
        key=lambda match: (
            -match["priority"],
            match["page_number"],
        )
    )

    return matches



if __name__ == "__main__":
    # Test using the existing indexes for Slide_2.pdf and Slide_3.pdf.
    document_ids = [
        "f99e09843ce822a7",
        "f37229f479daa345",
    ]

    try:
        matches = compare_document_pages(
            document_ids=document_ids,
            top_k=10,
            min_similarity=0.55,
        )

        if not matches:
            print(
                "\nNo page pairs met the similarity threshold. "
                "Try a lower threshold or inspect the page text."
            )

        for match in matches:
            print(
                f"\nDocument A: {match['document_a']}"
                f"\nPage A: {match['page_a']}"
                f"\nDocument B: {match['document_b']}"
                f"\nPage B: {match['page_b']}"
                f"\nSimilarity: {match['similarity']:.3f}"
                f"\nText A: {match['text_a'][:200]}"
                f"\nText B: {match['text_b'][:200]}"
            )

        print(f"\nTotal matching page pairs: {len(matches)}")

    except Exception as error:
        print(f"\nError: {error}")
