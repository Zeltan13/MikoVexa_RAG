import os
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag_pipeline import load_pdf_data, load_sql_data, retrieve_and_rerank
from model_loader import groq_api_key, load_bm25_index

#Load environment variables
load_dotenv()

#Load data
all_texts = load_pdf_data("Panduan_Lengkap_Pengguna_MikoVexa_AI_Update.pdf") + load_sql_data("mikovexa_synthetic_data.db")
bm25 = load_bm25_index(all_texts)

def chat_with_mikovexa(history, query):
    if not query:
        return history, "No query provided"

    retrieved_context = retrieve_and_rerank(query, all_texts, bm25)

    #LLM Call
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)
    prompt = f"""
    Anda adalah asisten AI MikoVexa.
    Tugas Anda adalah memberikan jawaban akurat berdasarkan informasi yang tersedia di dokumen dan database.  
    - Jangan menebak jawaban. Jika informasi tidak tersedia, katakan dengan jelas bahwa jawaban tidak ditemukan.  
    - Gunakan format daftar jika terdapat langkah-langkah atau banyak detail.  
    - Jika jawaban tidak langsung tersedia, coba gunakan informasi terkait atau alternatif.
    - Jawab dalam bahasa Indonesia
    
    Konteks yang ditemukan: 
    {retrieved_context}

    Pertanyaan: {query}

    Jawaban:
    """
    response = llm.invoke(prompt)
    response_text = response.content
    history.append((query, response_text))
    return history, ""

#Gradio Chat Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 MikoVexa AI Chatbot</h1>")

    chatbot = gr.Chatbot(label="MikoVexa AI Chat")
    query_input = gr.Textbox(placeholder="Tanyakan sesuatu...", show_label=False)
    submit_button = gr.Button("Kirim")

    submit_button.click(chat_with_mikovexa, inputs=[chatbot, query_input], outputs=[chatbot, query_input])
    query_input.submit(chat_with_mikovexa, inputs=[chatbot, query_input], outputs=[chatbot, query_input])

if __name__ == "__main__":
    demo.launch(debug=True)
