
# DocuMind AI — Agentic Document Intelligence Platform

An AI-powered document intelligence platform that enables users to extract information from PDFs and text-containing images and answer questions using OCR, semantic search, and Retrieval-Augmented Generation (RAG).

## Overview

DocuMind AI transforms unstructured documents into searchable knowledge. Users can upload a document, ask natural-language questions, and receive context-grounded answers with supporting source excerpts.

The system combines document ingestion, text extraction, vector-based retrieval, and large language models to make document understanding more accessible and efficient.

## Key Features

- **Multi-format Document Input:** Process PDF files and images containing readable text.
- **OCR-Based Text Extraction:** Use PaddleOCR to extract text from images.
- **Semantic Search:** Retrieve relevant document passages using Sentence Transformers and FAISS.
- **RAG-Based Question Answering:** Generate answers using retrieved document context and an OpenAI language model.
- **Source-Grounded Responses:** Display supporting source excerpts to help users verify answers.
- **Hallucination Reduction:** Instruct the model to answer from available document evidence and abstain when sufficient information cannot be found.
- **Document Identification:** Generate document IDs to identify uploaded documents within the application.
- **Interactive Interface:** Upload documents and ask questions through a Streamlit web application.
- **Evaluation Pipeline:** Test retrieval, answer quality, citation correctness, numeric answers, and abstention behavior.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| OCR | PaddleOCR |
| PDF Processing | PyMuPDF |
| Text Embeddings | Sentence Transformers |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| RAG Framework | LangChain |
| Language Model Integration | LangChain OpenAI |
| LLM | OpenAI `gpt-4o-mini` |
| Evaluation | Custom Python evaluation pipeline |

## System Architecture

```text
          PDF / Image Upload
                  |
                  v
         Document Ingestion
                  |
                  v
       Text Extraction / OCR
                  |
                  v
           Text Chunking
                  |
                  v
       Sentence Transformers
                  |
                  v
          FAISS Vector Index
                  |
                  v
       Semantic Retrieval
                  |
                  v
      Retrieved Document Context
                  |
                  v
       LangChain + OpenAI LLM
                  |
                  v
       Grounded Answer + Sources
```

## How It Works

1. **Document ingestion:** The user uploads a PDF or an image containing readable text.
2. **Text extraction:** The application extracts text from PDFs or applies OCR to images.
3. **Chunking:** Extracted text is divided into smaller passages for retrieval.
4. **Embedding generation:** Sentence Transformers converts passages into numerical vector representations.
5. **Indexing and retrieval:** FAISS searches for passages semantically relevant to the user's question.
6. **Answer generation:** LangChain supplies retrieved context to the OpenAI language model.
7. **Evidence presentation:** The application returns an answer with supporting source excerpts when available.
8. **Evaluation:** A custom test pipeline checks retrieval and answer behavior against defined test cases.

## Project Structure

```text
DocuMind-AI/
├── app.py
├── requirements.txt
├── rebuild_indexes.py
├── .gitignore
├── ingestion/
│   └── document_loader.py
├── retrieval/
│   ├── chunking.py
│   ├── rag.py
│   └── semantic_search.py
└── evaluation/
    ├── config.json
    ├── evaluate.py
    ├── relevance.py
    └── test_dataset.json
```

## Installation and Setup

### Prerequisites

- Python 3.10
- Git
- An OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/SaraJain27/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own API key. Never commit `.env` or publish your API key.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Evaluation

The project includes a custom evaluation pipeline with a test dataset.

The latest local evaluation run completed with the following results:

| Metric | Result |
|---|---:|
| Total test cases | 11 |
| Completed tests | 11/11 |
| Evaluation errors | 0 |
| Page retrieval | 9/9 |
| Citation page correctness | 9/9 |
| Abstention classification accuracy | 11/11 |
| Correct refusals on unanswerable tests | 2/2 |
| Unexpected refusals on answerable tests | 0/9 |
| Expected numeric values present | 4/4 |

These results represent the current local test suite and are not a guarantee of performance on every document or question.

To run the evaluation:

```bash
python -m evaluation.evaluate
```

The evaluation script may create a local results file. Review its contents before sharing any evaluation output publicly.

## Reliability Considerations

- Answers are generated using retrieved document context.
- The application is designed to abstain when available evidence is insufficient.
- Source excerpts help users verify the information used to generate answers.
- Evaluation covers retrieval behavior, source-page correctness, numeric answers, and unanswerable questions.

LLM-generated answers and OCR output can still contain errors. Important information should be verified against the original document.

## Security and Privacy

- Store API keys in environment variables.
- Do not commit `.env`, uploaded private documents, or sensitive document metadata.
- Review document-handling and storage behavior before deploying the application for other users.
- Use authorized documents and protect confidential information.

## Future Improvements

- Add support for more document formats.
- Improve retrieval ranking and document-level filtering.
- Expand evaluation with larger, more diverse test datasets.
- Add automated regression testing for document extraction and question answering.
- Improve deployment, observability, and document lifecycle management.

## Author

**Sara Jain**

GitHub: [@SaraJain27](https://github.com/SaraJain27)

Project Repository: [DocuMind AI](https://github.com/SaraJain27/DocuMind-AI)

---

*DocuMind AI — Turning documents into searchable, evidence-grounded knowledge.*
