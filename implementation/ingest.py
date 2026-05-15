import os, glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

def fetch_documents():
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(documents)

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)

if __name__ == "__main__":
    create_embeddings(create_chunks(fetch_documents()))
    print("Ingestion complete")
