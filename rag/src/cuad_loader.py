import json
from pathlib import Path

from langchain_core.documents import Document


class CUADLoader:
    """
    Loads the CUAD legal knowledge base into LangChain Documents.
    CUAD's CSV / JSON / TXT are already clean structured text, so they are
    read directly (no OCR). Two ways:
        1. from a local cuad_sample.json  (quick demo)
        2. from HuggingFace (theatticusproject/cuad)  (full dataset)
    """

    def __init__(self, sample_path: str = None):
        self.sample_path = Path(sample_path) if sample_path else None

    def load_sample(self):
        documents = []

        try:
            print(f"Loading CUAD sample: {self.sample_path}")
            with open(self.sample_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            for idx, rec in enumerate(records, start=1):
                doc = Document(
                    page_content=rec["text"],
                    metadata={
                        "source_file": "CUAD",
                        "source_name": rec.get("type", "contract"),
                        "source_index": idx,
                    },
                )
                documents.append(doc)

            print("CUAD sample loaded")

        except Exception as e:
            print(f"Error loading CUAD sample: {e}")
            return

        return documents

    def load_from_huggingface(self, limit: int = 100):
        documents = []

        try:
            from datasets import load_dataset
            print("Downloading CUAD from HuggingFace (theatticusproject/cuad)...")
            ds = load_dataset("theatticusproject/cuad", split="train")

            seen = set()
            idx = 0
            for row in ds:
                context = row.get("context", "")
                title = row.get("title", "")
                if context and title not in seen:
                    seen.add(title)
                    idx += 1
                    doc = Document(
                        page_content=context[:4000],
                        metadata={
                            "source_file": "CUAD",
                            "source_name": title or "contract",
                            "source_index": idx,
                        },
                    )
                    documents.append(doc)
                if idx >= limit:
                    break

            print(f"CUAD loaded from HuggingFace ({len(documents)} contracts)")

        except ImportError:
            print("Install first:  pip install datasets")
            return
        except Exception as e:
            print(f"Error loading CUAD from HuggingFace: {e}")
            return

        return documents
