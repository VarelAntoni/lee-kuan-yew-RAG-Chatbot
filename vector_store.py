import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import fungsi dari file step 1 yang sudah kamu buat
from data_ingestion import load_documents, split_documents

DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    print("1. Memulai proses Data Ingestion...")
    docs = load_documents()
    chunks = split_documents(docs)
    
    print("\n2. Mengunduh/Memuat Model Embedding (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    print("\n3. Mengubah teks menjadi vektor dan menyimpannya ke FAISS Database...")
    print("Proses ini mungkin memakan waktu beberapa saat tergantung jumlah PDF...")
    
    # Membuat database FAISS dari chunks teks
    db = FAISS.from_documents(chunks, embeddings)
    
    # Membuat direktori jika belum ada dan menyimpan database
    os.makedirs('vectorstore', exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    
    print(f"\nBerhasil! Vector Database telah disimpan di folder: '{DB_FAISS_PATH}'")

if __name__ == "__main__":
    create_vector_db()