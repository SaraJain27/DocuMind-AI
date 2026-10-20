import hashlib
import json
import re
from pathlib import Path

import streamlit as st

from ingestion.document_loader import extract_image_text, load_pdf
from retrieval.semantic_search import build_index
from retrieval.rag import answer_question_with_sources, compare_documents


st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# Appearance settings
st.sidebar.subheader("🎨 Customize Appearance")
themes = {
    "Soft Sage": {"background": "#F4F6F4", "text": "#26332B", "sidebar": "#E9EEE9"},
    "Soft Blue Grey": {"background": "#F2F4F8", "text": "#2D3442", "sidebar": "#E8ECF3"},
    "Warm Cream": {"background": "#F7F4EF", "text": "#38352F", "sidebar": "#EEE9E0"},
}
theme_name = st.sidebar.selectbox("Choose Background Theme", list(themes.keys()), key="background_theme")
selected_theme = themes[theme_name]
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{ background: {selected_theme["background"]} !important; }}
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{ background: transparent !important; }}
    [data-testid="stSidebar"] {{ background: {selected_theme["sidebar"]} !important; }}
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{ color: {selected_theme["text"]} !important; }}
    [data-testid="stFileUploader"], [data-testid="stForm"] {{ border-radius: 12px; }}
    [data-testid="stAppViewContainer"] {{ transition: background 0.3s ease; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 DocuMind AI")
st.subheader("Ask questions about your documents")
st.write("Upload PDFs or images, ask questions, and compare information across documents.")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
REGISTRY_PATH = PROJECT_ROOT / "data" / "documents.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def read_registry():
    """Load previously registered documents."""
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(documents):
    """Save registered documents."""
    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)


def process_document(uploaded_file):
    """Save a document and create its index if necessary."""
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    extension = Path(uploaded_file.name).suffix.lower()
    file_path = UPLOAD_DIR / f"{document_id}{extension}"

    if not file_path.exists():
        file_path.write_bytes(file_bytes)

    documents = read_registry()
    if document_id not in documents:
        documents[document_id] = {
            "document_id": document_id,
            "filename": uploaded_file.name,
            "file_path": str(file_path),
            "extension": extension,
        }
    else:
        documents[document_id]["filename"] = uploaded_file.name
        documents[document_id]["file_path"] = str(file_path)
        documents[document_id]["extension"] = extension
    save_registry(documents)

    index_path = INDEX_DIR / f"{document_id}.faiss"
    metadata_path = INDEX_DIR / f"{document_id}.json"

    if not index_path.exists() or not metadata_path.exists():
        if extension == ".pdf":
            pages = load_pdf(str(file_path))
        else:
            items = extract_image_text(str(file_path))
            pages = [{
                "document_name": uploaded_file.name,
                "page_number": 1,
                "text": "\n".join(item.get("text", "") for item in items),
            }]

        for page in pages:
            page["document_name"] = uploaded_file.name

        if not any(page.get("text", "").strip() for page in pages):
            raise ValueError(f"No readable text found in {uploaded_file.name}.")

        build_index(pages, document_id=document_id)

    return document_id



def extract_source_excerpt(text, topic, question=""):
    """Return a concise, question-focused source excerpt for display."""
    if not text:
        return ""

    cleaned = " ".join(str(text).split()).strip()

    if topic == "invoice":
        q = question.lower()

        # Match invoice labels and their corresponding amounts.
        amount = r"\$?\s*[\d,]+(?:\.\d{2})"

        if "subtotal" in q:
            patterns = [
                rf"\bSUBTOTAL\s*[:\-]?\s*{amount}",
            ]
        elif "tax rate" in q:
            patterns = [
                r"\bTAX\s+RATE\s*[:\-]?\s*\d+(?:\.\d+)?\s*%?"
            ]
        elif "tax" in q:
            patterns = [
                rf"\bTAX\s*[:\-]?\s*{amount}",
            ]
        elif any(term in q for term in (
            "final total",
            "invoice total",
            "total amount",
            "amount due",
            "what is the total",
            "total",
        )):
            patterns = [
                rf"\bOTHER\s+TOTAL\s*[:\-]?\s*{amount}",
                rf"\bTOTAL\s+PAYMENT\s+DUE\s*[:\-]?\s*{amount}",
                rf"\bTOTAL\s+DUE\s*[:\-]?\s*{amount}",
            ]
        else:
            patterns = [
                rf"\bSUBTOTAL\s*[:\-]?\s*{amount}",
                rf"\bTAX\s*[:\-]?\s*{amount}",
                rf"\bOTHER\s+TOTAL\s*[:\-]?\s*{amount}",
            ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # Never display an unrelated OCR excerpt as if it were evidence.
        return ""

    section_starts = {
        "education": ("education",),
        "skills": ("technical skills",),
        "projects": (
            "skypath",
            "documind ai",
            "pneumonia detection from chest x-rays",
            "pneumonia detection",
        ),
    }

    section_ends = {
        "education": (
            "projects", "technical skills", "certifications", "leadership"
        ),
        "skills": (
            "certifications", "professional development", "leadership"
        ),
        "projects": (
            "technical skills", "certifications",
            "professional development", "leadership"
        ),
    }

    lower = cleaned.lower()
    starts = section_starts.get(topic, ())
    positions = [lower.find(term) for term in starts if lower.find(term) >= 0]

    if positions:
        start_position = min(positions)
        cleaned = cleaned[start_position:]
        lower = cleaned.lower()

    ends = section_ends.get(topic, ())
    end_positions = [lower.find(term) for term in ends if lower.find(term) > 0]

    if end_positions:
        cleaned = cleaned[:min(end_positions)].strip(" |•-;,.")

    return cleaned


# Upload interface
st.divider()
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF or image files",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    processed_documents = {}
    for uploaded_file in uploaded_files:
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                document_id = process_document(uploaded_file)
            processed_documents[document_id] = uploaded_file.name
            st.success(f"Ready: {uploaded_file.name}")
        except Exception as error:
            st.error(f"Could not process {uploaded_file.name}: {error}")

    if processed_documents:
        st.session_state["current_documents"] = processed_documents


# Chat interface
current_documents = st.session_state.get("current_documents", {})

if current_documents:
    st.divider()
    st.subheader("💬 Chat with Your Documents")
    selected_ids = st.multiselect(
        "Select documents for this chat",
        options=list(current_documents.keys()),
        format_func=lambda doc_id: current_documents[doc_id],
        default=list(current_documents.keys()),
        key="chat_selected_documents",
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_document_ids" not in st.session_state:
        st.session_state.chat_document_ids = []

    if selected_ids != st.session_state.chat_document_ids:
        st.session_state.chat_messages = []
        st.session_state.chat_document_ids = list(selected_ids)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View source pages"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['filename']} — Page {source['page_number']}**"
                        )
                        st.caption(source["text"])

    question = st.chat_input("Ask anything about your selected documents...")

    if question:
        if not selected_ids:
            st.warning("Please select at least one document.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Reading your documents..."):
                    answer = ""
                    sources = []
                    try:
                        if len(selected_ids) == 1:
                            # Generate the answer and return the exact evidence
                            # used by RAG. Do not run a second, independent search
                            # to populate the source panel.
                            document_id = selected_ids[0]
                            rag_result = answer_question_with_sources(
                                question,
                                document_id=document_id,
                            )
                            answer = str(rag_result.get("answer", ""))
                            evidence_sources = rag_result.get("sources", [])
                            filename = current_documents.get(document_id, document_id)

                            normalized_question = " ".join(
                                question.lower().strip().split()
                            )
                            if any(term in normalized_question for term in (
                                "educational qualification", "education", "degree",
                                "cgpa", "university", "qualification",
                            )):
                                topic = "education"
                            elif any(term in normalized_question for term in (
                                "technical skills", "what skills", "which skills",
                                "programming languages", "technologies", "tools mentioned",
                            )):
                                topic = "skills"
                            elif any(term in normalized_question for term in (
                                "what projects", "which projects", "list the projects",
                                "projects are listed", "project names", "name the projects",
                            )):
                                topic = "projects"
                            elif any(term in normalized_question for term in (
                                "subtotal", "tax amount", "tax rate", "final total",
                                "invoice total", "what is the total", "amount due",
                            )):
                                topic = "invoice"
                            else:
                                topic = None

                            # Prefer evidence chunks that directly contain the requested
                            # invoice field. For invoice questions, show one concise source
                            # excerpt per page to avoid duplicate Page 1 entries.
                            evidence_items = list(evidence_sources)
                            if topic == "invoice":
                                question_terms = [
                                    term for term in (
                                        "subtotal", "tax rate", "tax", "other total",
                                        "total payment due", "amount due", "total",
                                    ) if term in normalized_question
                                ]

                                def invoice_relevance(item):
                                    text = str(item.get("text", "") or "").lower()
                                    return sum(1 for term in question_terms if term in text)

                                evidence_items.sort(key=invoice_relevance, reverse=True)

                            seen_sources = set()
                            seen_invoice_pages = set()
                            for item in evidence_items:
                                raw_text = str(item.get("text", "") or "")
                                page_number = item.get("page_number")
                                if not raw_text.strip():
                                    continue

                                if topic == "invoice" and page_number in seen_invoice_pages:
                                    continue

                                if topic:
                                    source_text = extract_source_excerpt(
                                        raw_text, topic, question=question
                                    )
                                    if not source_text:
                                        source_text = " ".join(raw_text.split()).strip()
                                else:
                                    source_text = " ".join(raw_text.split()).strip()

                                source_key = (page_number, source_text)
                                if source_key in seen_sources:
                                    continue
                                seen_sources.add(source_key)
                                if topic == "invoice":
                                    seen_invoice_pages.add(page_number)

                                sources.append({
                                    "filename": filename,
                                    "page_number": page_number,
                                    "text": source_text[:300] + ("..." if len(source_text) > 300 else ""),
                                })
                        else:
                            # Multi-document comparison uses the selected documents.
                            documents_to_compare = []
                            registry = read_registry()
                            for document_id in selected_ids:
                                saved = registry.get(document_id, {})
                                documents_to_compare.append({
                                    "document_id": document_id,
                                    "filename": current_documents.get(
                                        document_id,
                                        saved.get("filename", document_id),
                                    ),
                                })
                            answer = compare_documents(question, documents_to_compare)
                            st.caption(
                                "Multi-document comparison is generated from retrieved evidence "
                                "for each selected document."
                            )

                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 View source pages"):
                                for source in sources:
                                    page_label = (
                                        f"Page {source['page_number']}"
                                        if source.get("page_number") is not None
                                        else "Page not identified"
                                    )
                                    st.markdown(f"**{source['filename']} — {page_label}**")
                                    st.caption(source["text"])
                        elif len(selected_ids) == 1:
                            st.info(
                                "No source chunks were returned with this answer. "
                                "The assistant may not have found sufficient evidence."
                            )

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    except Exception as error:
                        st.error(f"Could not answer: {error}")

    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()
else:
    st.info("Upload at least one document to start chatting.")
