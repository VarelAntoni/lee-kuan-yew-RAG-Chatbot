import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Tentukan direktori dataset
DATA_PATH = "dataset/"

def load_documents():
    print("Membaca dokumen dari direktori...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Total halaman yang dimuat: {len(documents)}")
    return documents

def split_documents(documents):
    print("Memecah teks menjadi chunks...")
    # LKY sering bicara dalam paragraf panjang, kita set chunk agak besar
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Eksekusi Step 1
    docs = load_documents()
    chunks = split_documents(docs)
    
    # Preview hasil untuk memastikan metadata siap untuk Evaluasi nanti
    if len(chunks) > 0:
        print(f"\nBerhasil membuat {len(chunks)} chunks.")
        print("Preview Chunk Pertama:")
        print("-" * 50)
        print(chunks[0].page_content[:300] + "...")
        print("-" * 50)
        print(f"Metadata (Penting untuk Eval): {chunks[0].metadata}")
    else:
        print("Tidak ada chunks yang dibuat. Pastikan PDF ada di folder dataset/")