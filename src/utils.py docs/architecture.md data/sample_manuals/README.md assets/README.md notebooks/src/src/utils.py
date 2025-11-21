import os
from langchain.document_loaders import PyPDFLoader

def load_pdfs_from_dir(data_dir: str):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            docs.extend(pages)
    return docs
