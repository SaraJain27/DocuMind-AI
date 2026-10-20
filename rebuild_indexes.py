
import json
import shutil
from pathlib import Path

from ingestion.document_loader import extract_image_text, load_pdf
from retrieval.semantic_search import build_index, get_index_paths


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "documents.json"
BACKUP_DIR = PROJECT_ROOT / "data" / "indexes_backup_before_rechunking"


def main():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Document registry not found: {REGISTRY_PATH}"
        )

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, dict):
        raise ValueError("documents.json must contain a JSON object.")

    print("=" * 60)
    print("DocuMind AI - Index Rebuild")
    print("=" * 60)
    print("Registered documents:", len(documents))
    print("Backup directory:", BACKUP_DIR)
    print()

    succeeded = []
    failed = []

    for document_id, metadata in documents.items():
        filename = metadata["filename"]
        file_path = Path(metadata["file_path"])

        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path

        print(f"\nProcessing: {filename}")
        print("Document ID:", document_id)

        try:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Original file not found: {file_path}"
                )

            extension = file_path.suffix.lower()

            if extension == ".pdf":
                pages = load_pdf(str(file_path))
            elif extension in {".png", ".jpg", ".jpeg"}:
                items = extract_image_text(str(file_path))
                pages = [{
                    "document_name": filename,
                    "page_number": 1,
                    "text": "\n".join(
                        item["text"] for item in items
                    ),
                }]
            else:
                raise ValueError(
                    f"Unsupported file extension: {extension}"
                )

            for page in pages:
                page["document_name"] = filename

            if not any(
                page.get("text", "").strip() for page in pages
            ):
                raise ValueError("No readable text found.")

            # Save the existing index and metadata before replacing them.
            index_path, metadata_path = get_index_paths(document_id)

            for path in (index_path, metadata_path):
                if path.exists():
                    backup_path = BACKUP_DIR / path.name
                    if not backup_path.exists():
                        shutil.copy2(path, backup_path)

            chunk_count = build_index(
                pages,
                document_id=document_id,
            )

            print("Pages extracted:", len(pages))
            print("Chunks indexed:", chunk_count)
            print("Status: SUCCESS")

            succeeded.append(document_id)

        except Exception as error:
            print("Status: FAILED")
            print("Reason:", error)
            failed.append({
                "document_id": document_id,
                "filename": filename,
                "error": str(error),
            })

    print("\n" + "=" * 60)
    print("REBUILD SUMMARY")
    print("=" * 60)
    print("Successful:", len(succeeded))
    print("Failed:", len(failed))

    for item in failed:
        print(
            f"- {item['filename']} ({item['document_id']}): "
            f"{item['error']}"
        )

    print("\nOriginal documents and registry were not modified.")
    print("Backup remains at:", BACKUP_DIR)


if __name__ == "__main__":
    main()
