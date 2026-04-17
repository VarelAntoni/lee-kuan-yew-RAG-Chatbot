import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from chatbot_engine import get_lky_response

load_dotenv()

# 1. Konfigurasi LLM sebagai "Juri Evaluator"
evaluator_llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0 
)

# 2. Merancang Prompt Evaluasi
eval_prompt = ChatPromptTemplate.from_template("""
You are an impartial and strict AI evaluator. 
Your task is to evaluate a RAG (Retrieval-Augmented Generation) chatbot based on two metrics.

METRICS:
1. Faithfulness (Score 1-5): Is the Answer supported entirely by the provided Context? If the Answer contains facts not in the Context, give a lower score.
2. Relevance (Score 1-5): Does the Answer directly address the user's Question?

DATA TO EVALUATE:
Question: {question}
Context: {context}
Answer: {answer}

OUTPUT FORMAT (Strictly follow this):
Faithfulness: [Score]/5
Relevance: [Score]/5
Reasoning: [1-2 short sentences explaining the scores]
""")

eval_chain = eval_prompt | evaluator_llm

def run_evaluation():
    print("Memulai Proses Evaluasi Otomatis (LLM-as-a-Judge)...\n")
    
    # 3. Menyiapkan Test Cases (Pertanyaan Ujian)
    test_questions = [
        "What is your view on talent and meritocracy in Singapore?",
        "How should a small country survive geopolitically?",
        "What is the best food in Singapore?" # Pertanyaan jebakan (harus dijawab 'tidak tahu' oleh LKY)
    ]
    
    total_faithfulness = 0
    total_relevance = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"--- Ujian {i}/{len(test_questions)} ---")
        print(f"Q: {question}")
        
        # A. Dapatkan jawaban dari Chatbot LKY
        print("Sedang mengambil jawaban dari LKY Chatbot...")
        rag_result = get_lky_response(question)
        answer = rag_result['answer']
        
        # Menggabungkan semua teks dari dokumen sumber menjadi satu string konteks
        context_text = "\n".join([doc.page_content for doc in rag_result['context']])
        
        # B. Kirim ke Juri Evaluator
        print("Sedang dievaluasi oleh LLM Judge...")
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text,
            "answer": answer
        })
        
        # C. Tampilkan Hasil
        print(f"LKY Answer: {answer}")
        print(f"\n[EVALUATION RESULT]")
        print(eval_result.content)
        print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()