
import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from retrieval.semantic_search import (
    search_documents,
    compare_document_pages,
    find_exact_page_matches,
)


# Load environment variables from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Check your .env file."
    )


# Create the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the user's question about an uploaded document.

Return exactly one label:
overview
specific_question
page_lookup

Use overview when the user wants a summary, explanation,
description, purpose, main topics, or general understanding
of the whole document.

Use page_lookup when the user asks where something appears
or asks for a page number.

Use specific_question for questions about particular facts,
details, values, people, dates, or sections.

Return only the label."""
    ),
    (
        "human",
        "User question: {question}"
    ),
])


def classify_question(question: str) -> str:
    """Classify the intent of a document question."""

    try:
        messages = intent_prompt.format_messages(
            question=question
        )
        response = llm.invoke(messages)
        intent = response.content.strip().lower()

        if intent in {
            "overview",
            "specific_question",
            "page_lookup",
        }:
            return intent

    except Exception:
        pass

    # Safe fallback if classification fails.
    return "specific_question"

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
      """You are DocuMind AI, a document understanding assistant.

Use only the supplied document evidence.

Rules:
- Answer the user's question directly and concisely.
- Never invent facts, fees, explanations, or conclusions.
- Preserve all relevant factual details present in the evidence, especially dates,
  date ranges, names, amounts, identifiers, and numerical values.
- When the evidence contains a date range (for example, 2022–2025), include
  both the start year and end year. Do not shorten a date range to only its
  ending year.
- For education questions, include the degree, institution, CGPA or grade,
  and complete study period whenever those details are present in the evidence.
- For amounts, reproduce the exact value shown in the document.
- Do not infer shipping, handling, or other charges unless the evidence explicitly
  identifies them.
- Mention source page numbers only when provided by the evidence.
- If evidence is insufficient, clearly state that.
- Do not add explanations that are not supported by the document.
- Before finalizing, check that the answer preserves the relevant dates and
  numerical details from the evidence without adding unsupported facts."""
    ),
    (
        "human",
        """Document evidence:
{context}

User question:
{question}

Answer using only the evidence above."""
    ),
])




def answer_question_with_sources(
    question: str,
    document_id: str = "default",
) -> dict:
    """Return the generated answer together with the evidence used."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    normalized_question = question.lower().strip()
    intent = classify_question(question)
    print(f"DEBUG question intent: {intent}")

    # Handle questions asking for a page number.
    page_keywords = (
        "what page",
        "which page",
        "on what page",
        "on which page",
        "find page",
        "page number",
    )

    if any(keyword in normalized_question for keyword in page_keywords):
        exact_matches = find_exact_page_matches(question, document_id)

        if exact_matches:
            best_match = max(
                exact_matches,
                key=lambda match: (
                    match.get("priority", 0),
                    -match["page_number"],
                ),
            )

            answer = (
                f"**Answer:** The topic appears on **Page "
                f"{best_match['page_number']}**.\n\n"
                f"**Source:** {best_match['text'][:300]}"
            )
            return {
                "answer": answer,
                "sources": [{
                    "page_number": best_match.get("page_number"),
                    "text": best_match.get("text", ""),
                }],
            }

    # Decide whether this is an overview question.
    
    is_overview_question = intent == "overview"
    retrieved_results = []

    # Detect questions asking about resume or document projects.
    project_question_terms = (
        "what projects",
        "which projects",
        "list the projects",
        "projects are listed",
        "project names",
        "name the projects",
    )

    is_project_question = any(
        term in normalized_question
        for term in project_question_terms
    )


    if is_overview_question:
        # Read the selected document's saved metadata.
        metadata_path = "data/indexes/" + document_id + ".json"

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, json.JSONDecodeError):
            metadata = []

        # Prefer the first page, introduction and conclusion.
        priority_terms = (
            "introduction",
            "background and context",
            "problem statement",
            "project objectives",
            "summary of work",
            "conclusion",
            "abstract",
            "executive summary",
        )

        selected = []
        seen = set()

        if isinstance(metadata, list):
            # The first page often contains the title and document type.
            first_pages = [
                item
                for item in metadata
                if item.get("page_number") == 1
            ]
            selected.extend(first_pages[:2])

            # Add passages with overview-relevant headings.
            for item in metadata:
                text = item.get("text", "")
                lowered = text.lower()

                if any(term in lowered for term in priority_terms):
                    selected.append(item)

        # Add semantic search results as supplementary evidence.
        overview_queries = (
            "project introduction background and problem statement",
            "project objectives and purpose",
            "main findings and conclusion",
        )

        for search_query in overview_queries:
            results = search_documents(
                search_query,
                top_k=3,
                document_id=document_id,
                min_similarity=0.20,
            )
            selected.extend(results)

        # Deduplicate and limit the context size.
        for item in selected:
            text = item.get("text", "").strip()
            page = item.get("page_number")

            if not text:
                continue

            key = (page, text)

            if key not in seen:
                seen.add(key)
                retrieved_results.append({
                    "page_number": page,
                    "text": text,
                })

        retrieved_results = retrieved_results[:10]

    
    else:
        if is_project_question:
            # Read saved chunks and prioritize the PROJECTS section.
            metadata_path = "data/indexes/" + document_id + ".json"

            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
            except (OSError, json.JSONDecodeError):
                metadata = []

            project_chunks = []
            seen_chunks = set()

            if isinstance(metadata, list):
                for item in metadata:
                    text = item.get("text", "").strip()
                    lowered = text.lower()

                    # Keep chunks that contain project evidence.
                    has_project_heading = "projects" in lowered
                    has_known_project = any(
                        name in lowered
                        for name in (
                            "flight booking assistant",
                            "documind ai",
                            "pneumonia detection",
                        )
                    )

                    if has_project_heading or has_known_project:
                        key = (
                            item.get("page_number"),
                            text,
                        )

                        if text and key not in seen_chunks:
                            seen_chunks.add(key)
                            project_chunks.append({
                                "page_number": item.get("page_number"),
                                "text": text,
                            })

            retrieved_results = project_chunks[:8]

            # If the metadata did not provide project chunks,
            # fall back to normal semantic retrieval.
            if not retrieved_results:
                retrieved_results = search_documents(
                    question,
                    top_k=5,
                    document_id=document_id,
                    min_similarity=0.10,
                )

        else:
            # Existing retrieval for ordinary questions.
            retrieved_results = search_documents(
                question,
                top_k=5,
                document_id=document_id,
                min_similarity=0.25,
            )

            # Broader fallback when no chunks pass the threshold.
            if not retrieved_results:
                fallback_query = (
                    f"Relevant information, facts, values, amounts, "
                    f"labels and details related to: {question}"
                )

                retrieved_results = search_documents(
                    fallback_query,
                    top_k=5,
                    document_id=document_id,
                    min_similarity=0.10,
                )


    # Reject clearly unrelated evidence using the original question.
    # This is deliberately conservative so valid low-score invoice/resume
    # questions are not rejected solely because of a low embedding score.
    if intent != "overview" and not is_project_question:
        direct_results = search_documents(
            question,
            top_k=3,
            document_id=document_id,
            min_similarity=0.0,
        )

        best_direct_score = max(
            (
                float(item.get("similarity", 0.0))
                for item in direct_results
            ),
            default=0.0,
        )

        stop_words = {
            "what", "which", "where", "when", "who", "why",
            "how", "is", "are", "was", "were", "the", "a",
            "an", "of", "for", "to", "in", "on", "and", "or",
            "does", "do", "did", "me", "please", "tell",
        }

        query_terms = {
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9]+", question)
            if len(word) > 2 and word.lower() not in stop_words
        }

        direct_text = " ".join(
            item.get("text", "").lower()
            for item in direct_results
        )

        has_keyword_match = any(
            re.search(rf"\b{re.escape(term)}\b", direct_text)
            for term in query_terms
        )

        if best_direct_score < 0.15 and not has_keyword_match:
            return {
                "answer": (
                    "I could not find sufficient information in the "
                    "document to answer this question."
                ),
                "sources": [],
            }

    # Do not generate an answer without retrieved evidence.
    if not retrieved_results:
        return {
            "answer": (
                "I could not find sufficient information in the "
                "document to answer this question."
            ),
            "sources": [],
        }

    # Prepare evidence for the language model.
    context_parts = []

    for result in retrieved_results:
        context_parts.append(
            f"Page {result['page_number']}:\n{result['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    # Return the exact evidence supplied to the LLM and make its page numbers
    # visible in the answer for transparent, text-based evaluation. This is an
    # evidence list, not a claim that every listed page supports every sentence.
    sources = [
        {
            "page_number": result.get("page_number"),
            "text": result.get("text", ""),
        }
        for result in retrieved_results
    ]

    answer = str(response.content or "").strip()

    # Do not append evidence-page labels to answers that clearly abstain.
    lowered_answer = answer.lower()
    refusal_phrases = (
        "could not find sufficient information",
        "does not mention",
        "does not contain the information",
        "does not contain information",
        "not mentioned in the document",
        "not specified in the document",
        "not provided in the document",
        "not found in the document",
        "insufficient information",
        "cannot determine from the document",
        "can't determine from the document",
        "cannot answer based on the document",
        "i don't have enough information",
        "i do not have enough information",
        "evidence does not mention",
        "evidence does not contain",
    )
    is_refusal = any(phrase in lowered_answer for phrase in refusal_phrases)
        # Do not display irrelevant retrieved chunks for refused answers.
    if is_refusal:
        sources = []

    evidence_pages = sorted({
        int(source["page_number"])
        for source in sources
        if source.get("page_number") is not None
    })

    if evidence_pages and not is_refusal:
        page_label = "; ".join(f"Page {page}" for page in evidence_pages)
        answer += f"\n\n**Evidence retrieved from:** {page_label}."

    return {
        "answer": answer,
        "sources": sources,
    }



def answer_question(
    question: str,
    document_id: str = "default",
) -> str:
    """Backward-compatible function returning only the answer text."""
    result = answer_question_with_sources(question, document_id)
    return result["answer"]


def compare_documents(
    question: str,
    documents: list[dict],
) -> str:
    """Compare selected documents using relevant evidence."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if len(documents) < 2:
        raise ValueError(
            "Select at least two documents to compare."
        )

    # Get the document IDs and filenames from the selected documents.
    document_ids = [
        document["document_id"] for document in documents
    ]

    filename_by_id = {
        document["document_id"]: document["filename"]
        for document in documents
    }

    # Detect questions asking for matching or common topics.
    topic_keywords = (
        "same topic",
        "similar topic",
        "common topic",
        "matching topic",
        "same topics",
        "similar topics",
        "common topics",
        "matching topics",
        "which pages",
        "matching pages",
        "similar pages",
        "compare topics",
    )

    normalized_question = question.lower()

    if any(
        keyword in normalized_question
        for keyword in topic_keywords
    ):
        matches = compare_document_pages(
            document_ids=document_ids,
            top_k=10,
            min_similarity=0.55,
        )

        if not matches:
            return (
                "No sufficiently similar page pairs were found. "
                "Try a lower similarity threshold or check the "
                "extracted text in the selected PDFs."
            )

        match_lines = []

        for match in matches:
            filename_a = filename_by_id.get(
                match["document_a"],
                match["document_a"],
            )
            filename_b = filename_by_id.get(
                match["document_b"],
                match["document_b"],
            )

            match_lines.append(
                f"- {filename_a}, page {match['page_a']} "
                f"matches {filename_b}, page {match['page_b']} "
                f"(similarity: {match['similarity']:.3f}).\n"
                f"  First page content: {match['text_a']}\n"
                f"  Second page content: {match['text_b']}"
            )

        return (
            "Potential matching topics across the selected documents:\n\n"
            + "\n\n".join(match_lines)
            + "\n\nNote: Similarity scores indicate semantic "
              "closeness, not guaranteed identical topics."
        )

    # For other comparison questions, retrieve relevant chunks.
    evidence_sections = []

    for document in documents:
        document_id = document["document_id"]
        filename = document["filename"]

        results = search_documents(
            question,
            top_k=5,
            document_id=document_id,
            min_similarity=0.25,
        )

        if results:
            evidence = "\n\n".join(
                (
                    f"Page {result['page_number']} "
                    f"(similarity: {result['similarity']:.3f}):\n"
                    f"{result['text']}"
                )
                for result in results
            )
        else:
            evidence = (
                "No relevant text chunks were retrieved "
                "from this document."
            )

        evidence_sections.append(
            f"Document: {filename}\n"
            f"Evidence:\n{evidence}"
        )

    context = "\n\n---\n\n".join(evidence_sections)

    comparison_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You compare documents using only the evidence provided.

Rules:
- Consider evidence from every document separately.
- Answer the comparison question directly.
- Mention document filenames and page numbers when available.
- State numerical differences only when supported by evidence.
- Never invent missing values or assume missing information is zero.
- If the evidence is insufficient, clearly say so.
- Keep the answer concise and easy to understand."""
        ),
        (
            "human",
            """Question: {question}

Evidence from the selected documents:
{context}

Compare the documents and provide one combined answer."""
        ),
    ])

    messages = comparison_prompt.format_messages(
        question=question,
        context=context,
    )

    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":
    while True:
        question = input(
            "\nAsk a question (or type exit): "
        ).strip()

        if question.lower() == "exit":
            break

        try:
            answer = answer_question(question)
            print(f"\nAnswer: {answer}")
        except Exception as error:
            print(f"\nError: {error}")
