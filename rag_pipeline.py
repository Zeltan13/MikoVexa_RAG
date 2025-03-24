import os
import json
import sqlite3
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from model_loader import embedding_model, faiss_index, load_bm25_index, reranker

#Load PDF Data
def load_pdf_data(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n".join([page.page_content for page in docs])

    #Extract tables separately
    structured_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                structured_data.append(json.dumps(table))  #Store as JSON

    #Adaptive Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=300)
    pdf_chunks = text_splitter.split_text(full_text)
    all_texts = pdf_chunks + structured_data

    return all_texts

#Load SQL Data
def load_sql_data(db_path="mikovexa_synthetic_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()  
    tables = ["users", "mikovexa_robots", "commands", "security_logs"]
    sql_texts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()] 
        cursor.execute(f"SELECT * FROM {table} LIMIT 50")
        rows = cursor.fetchall()
        sql_texts.extend([json.dumps(dict(zip(columns, row))) for row in rows])
    conn.close()
    return sql_texts
sql_texts = load_sql_data()

#Hybrid Retrieval & Reranking
def retrieve_and_rerank(query, all_texts, bm25, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k * 2)
    faiss_results = [all_texts[i] for i in indices[0]]

    bm25_results = bm25.get_top_n(query.split(), all_texts, n=top_k)

    combined_results = list(set(faiss_results + bm25_results))
    
    #Apply reranking
    scores = reranker.predict([(query, doc) for doc in combined_results])
    reranked_results = [doc for _, doc in sorted(zip(scores, combined_results), reverse=True)]
    
    return "\n".join(reranked_results[:top_k])
