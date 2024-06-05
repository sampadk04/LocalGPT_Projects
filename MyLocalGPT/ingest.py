import os

import torch

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# for document_loaders: https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader

from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, DEVICE_TYPE

def load_single_document(file_path: str) -> Document:
    # loads single document from a file path

    loader = None

    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)

    if loader is None:
        raise ValueError(f"Unsupported file type: {file_path}")

    return loader.load()[0]

def load_documents(source_dir: str) -> list[Document]:
    # loads all documents from the source documents directory
    all_files = os.listdir(source_dir)

    # filter out files to only include .txt, .pdf, and .csv files
    all_files = [file_path for file_path in all_files if file_path.endswith(".txt") or file_path.endswith(".pdf") or file_path.endswith(".csv")]

    return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files]

# define the main function
def main():
    # load documents and split them into chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"Split the documents into {len(texts)} chunks")

    # create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE}
    )

    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
        )
    db.persist()
    db = None


if __name__ == "__main__":
    main()