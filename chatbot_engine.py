import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load Vector Database yang sudah dibuat di Step 2
DB_FAISS_PATH = 'vectorstore/db_faiss'
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# 2. Konfigurasi Chatbot (OpenRouter)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="nvidia/nemotron-3-super-120b-a12b:free", 
)

# 3. Merancang System Prompt (Persona Lee Kuan Yew)
system_prompt = (
    "You are Lee Kuan Yew, the founding father of Singapore. "
    "Use the following pieces of retrieved context to answer the question. "
    "Your tone must be pragmatic, direct, analytical, and unsentimental. "
    "Focus on logic, economic stability, and long-term strategy. "
    "If you don't know the answer based on the context, say that you don't have enough information from your records. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 4. Membuat Chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)

def get_lky_response(user_input):
    # Mengambil jawaban beserta sumbernya (penting untuk Evaluasi nanti)
    response = rag_chain.invoke({"input": user_input})
    return response

if __name__ == "__main__":
    query = "What is your view on talent and meritocracy in Singapore?"
    print(f"\nUser: {query}")
    
    result = get_lky_response(query)
    
    print("-" * 50)
    print(f"LKY Chatbot: {result['answer']}")
    print("-" * 50)
    print("\nSumber Dokumen yang Digunakan (Untuk Eval):")
    for doc in result['context']:
        print(f"- {doc.metadata['source']} (Halaman {doc.metadata['page']})")