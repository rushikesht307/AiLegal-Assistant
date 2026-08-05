from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, DirectoryLoader


class DocumentLoader:

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_documents(self):
        documents = []

        try:
            print(f"Loading: {self.folder_path}")

            documents += DirectoryLoader(self.folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
            documents += DirectoryLoader(self.folder_path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
            documents += DirectoryLoader(self.folder_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
            print("Documents loaded")

            for idx, doc in enumerate(documents, start=1):
                source = doc.metadata.get("source", str(self.folder_path))

                doc.metadata["source_file"] = Path(source).name
                doc.metadata["source_name"] = Path(source).stem
                doc.metadata["source_index"] = idx

        except Exception as e:
            print(f"Error loading {self.folder_path}: {e}")
            return

        return documents
