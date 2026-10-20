import glob
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DOC_DIR = "sample_docs"
INDEX_DIR = "faiss_index"


def load_and_split(doc_dir):
  chunks = []
  txt_files = glob.glob(os.path.join(doc_dir, "*.txt"))

  for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
      text = f.read()

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for p in paragraphs:
      chunks.append(Document(page_content=p, metadata={"source": file_path}))

  return chunks


def main():
  print("Loading and chunking documents...")
  chunks = load_and_split(DOC_DIR)
  print(f"Total chunks created: {len(chunks)}")

  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

  print("Indexing chunks into FAISS vector store...")
  vectorstore = FAISS.from_documents(chunks, embeddings)
  vectorstore.save_local(INDEX_DIR)
  print("Data ingestion completed successfully!")


if __name__ == "__main__":
  main()
