import streamlit as st
from chatbot_engine import get_lky_response

# Konfigurasi Halaman
st.set_page_config(page_title="LKY Chatbot", page_icon="🇸🇬", layout="centered")

st.title("🇸🇬 What Would Lee Kuan Yew Do?")
st.markdown("Ask any question and get a pragmatic, unsentimental answer based on Lee Kuan Yew's historical speeches and memoirs.")

# Inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kolom input untuk user
if prompt := st.chat_input("Ask LKY a question about geopolitics, economy, or life..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan respon LKY
    with st.chat_message("assistant"):
        with st.spinner("Analyzing pragmatically..."):
            try:
                # Memanggil mesin chatbot kamu
                result = get_lky_response(prompt)
                answer = result['answer']
                
                # Menampilkan jawaban
                st.markdown(answer)
                
                # [Opsional] Menampilkan sumber dokumen dalam dropdown yang elegan
                with st.expander("📚 View Document Sources"):
                    st.write("This answer is grounded in the following records:")
                    for doc in result['context']:
                        st.caption(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
                
                # Simpan ke history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")