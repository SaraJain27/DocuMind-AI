
import re


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split document pages into meaningful, overlapping text chunks.

    Preserves document name and original page number.
    Prefers sentence and paragraph boundaries over arbitrary cuts.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        # Normalize whitespace without removing paragraph boundaries.
        paragraphs = [
            re.sub(r"[ \t]+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        # Keep paragraphs together when possible.
        units = []
        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                units.append(paragraph)
            else:
                # Split long paragraphs at sentence boundaries.
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                current = ""

                for sentence in sentences:
                    if len(sentence) > chunk_size:
                        if current:
                            units.append(current)
                            current = ""

                        for start in range(0, len(sentence), step):
                            piece = sentence[start:start + chunk_size].strip()
                            if piece:
                                units.append(piece)
                    elif not current:
                        current = sentence
                    elif len(current) + 1 + len(sentence) <= chunk_size:
                        current += " " + sentence
                    else:
                        units.append(current)
                        current = sentence

                if current:
                    units.append(current)

        # Combine smaller units into overlapping chunks.
        current_chunk = ""

        for unit in units:
            if not current_chunk:
                current_chunk = unit
            elif len(current_chunk) + 2 + len(unit) <= chunk_size:
                current_chunk += "\n\n" + unit
            else:
                chunks.append({
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "text": current_chunk,
                })

                # Preserve some preceding context.
                tail = current_chunk[-overlap:].strip() if overlap else ""
                current_chunk = f"{tail}\n\n{unit}".strip() if tail else unit

                # If the overlap makes the new chunk too large, trim it.
                if len(current_chunk) > chunk_size:
                    current_chunk = current_chunk[-chunk_size:]

        if current_chunk:
            chunks.append({
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "text": current_chunk,
            })

    return chunks
