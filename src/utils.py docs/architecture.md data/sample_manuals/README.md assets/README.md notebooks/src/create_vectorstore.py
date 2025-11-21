import argparse
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from utils import load_pdfs_from_dir

def create_index(data_dir: str, index_path: str):
    print("Loading documents…")
    docs = load_pdfs_from_dir(data_dir)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    os.makedirs(index_path, exist_ok=True)
    save_path = os.path.join(index_path, "faiss_index")
    vectordb.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sample_manuals")
    parser.add_argument("--index_path", default="./faiss_index")
    args = parser.parse_args()
    create_index(args.data_dir, args.index_path)
