# 🇸🇬 LKY RAG Chatbot

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** built for the **99 Group Assessment (Section 3)**. It is designed to answer questions pragmatically and unsentimentally, based purely on the historical speeches, writings, and memoirs of Singapore's founding father, Lee Kuan Yew.

## ✨ Key Features
- **Document Retrieval:** Uses FAISS vector database to retrieve highly relevant context from local PDF documents.
- **Custom Persona:** Engineered system prompt to ensure the LLM adopts Lee Kuan Yew's pragmatic, meritocratic, and direct tone.
- **Interactive UI:** A clean, ChatGPT-like web interface built with Streamlit, complete with document source tracking.
- **Automated Evaluation:** Includes an `LLM-as-a-Judge` framework (`eval_chatbot.py`) to measure the chatbot's *Faithfulness* and *Answer Relevance*.

## 🛠️ Tech Stack
- **Framework:** LangChain (`langchain-core`, `langchain-community`)
- **LLM:** Meta Llama 3 / Nvidia Nemotron (via OpenRouter API)
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Frontend:** Streamlit

## 🚀 Setup & Installation

**1. Clone the repository**
git clone [https://github.com/your-username/lky-rag-chatbot.git](https://github.com/your-username/lky-rag-chatbot.git)
cd lky-rag-chatbot

**2. Create a virtual environment (Recommended using Conda)**
conda create -n lky_ai python=3.10 -y
conda activate lky_ai

**3. Install Dependencies**
pip install langchain langchain-community langchain-openai langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv streamlit torchvision

**4. Set up API Key**
Create a .env file in the root directory and add your OpenRouter API key:
OPENROUTER_API_KEY=your_openrouter_api_key_here

## 🎮 How to Run
**1. Run the Chatbot Interface (Streamlit)**
Start the interactive web application:
streamlit run app.py

**2. Run the Evaluation Script**
Test the RAG pipeline's accuracy and groundedness using the automated LLM-as-a-Judge script:
python eval_chatbot.py

## 📁 Project Structure
- data_ingestion.py : Script to load and chunk PDF documents.
- vector_store.py : Script to embed chunks and store them into FAISS.
- chatbot_engine.py : The core RAG pipeline (Retrieval + Generation).
- app.py : Streamlit frontend application.
- eval_chatbot.py : Automated evaluation framework using Langchain.
- dataset/ : Directory containing the source PDF files.
- vectorstore/ : Directory containing the saved FAISS local database.
