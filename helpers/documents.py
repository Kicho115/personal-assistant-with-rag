
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_dir: str = "data") -> list[Document]:
    """Loads documents from the personal data folders.

    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    docs = []

    for folder in ["emails", "notes", "sms", "calendar"]:
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    # TODO: checar si el calendario tiene mas metadatos
                    text = f.read()
                doc = Document(page_content=text, metadata={"source": file_path, "type": folder})
                docs.append(doc)

    return docs

def split_documents(docs: list[Document], chunk_size: int = 256, chunk_overlap: int = 36) -> list[Document]:
    """Splits documents into overlapping chunks.

    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """
    chunks = []
    for doc in docs:
        chunks.extend(RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents([doc]))
    return chunks

# Pa testear :v
if __name__ == "__main__":
    docs = load_documents("../data")
    print(f"Loaded {len(docs)} documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
