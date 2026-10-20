
from pathlib import Path

import pymupdf
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def extract_image_text(image_path: str) -> list[dict]:
    """Extract text and confidence scores from an image."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    results = ocr.predict(str(path))
    extracted = []

    for result in results:
        data = result.json
        if callable(data):
            data = data()

        payload = data.get("res", data)

        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])

        for i, text in enumerate(texts):
            extracted.append({
                "text": str(text),
                "confidence": (
                    float(scores[i])
                    if i < len(scores)
                    else None
                ),
            })

    return extracted


def load_pdf(pdf_path: str) -> list[dict]:
    """Extract text from digital and scanned PDF pages."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must be a PDF.")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            method = "pdf_text"

            # Use OCR for scanned pages with little embedded text.
            if len(text) < 30:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )

                image_path = (
                    path.parent
                    / f"{path.stem}_page_{page_number}.png"
                )

                try:
                    pixmap.save(str(image_path))
                    items = extract_image_text(str(image_path))
                    text = "\n".join(
                        item["text"] for item in items
                    )
                finally:
                    image_path.unlink(missing_ok=True)

                method = "paddleocr"

            pages.append({
                "document_name": path.name,
                "page_number": page_number,
                "text": text,
                "extraction_method": method,
            })

    return pages
