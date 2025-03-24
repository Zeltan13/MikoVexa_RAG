# **MikoVexa RAG Chatbot** 🤖📚  

A **Retrieval-Augmented Generation (RAG) chatbot** built using **Gradio**, **FAISS**, **BM25**, and **LangChain**, designed to interact with and retrieve information from structured and unstructured data sources.  

---

## **🚀 Features**
✅ **Semantic Search**: Uses **FAISS** and **BM25** for efficient document retrieval.  
✅ **Document Parsing**: Supports **PDF and DOCX** document loading for knowledge extraction.  
✅ **Chat Interface**: Built with **Gradio** for an interactive user experience.  
✅ **Language Support**: Uses **Sastrawi** for Indonesian text processing.  
✅ **Custom RAG Pipeline**: Combines **vector search + keyword search** for better accuracy.  

---

## 🛠 Installation & Setup
**1️⃣ Clone the repository**
```
git clone https://huggingface.co/spaces/zeltan13/MikoVexa
cd MikoVexa
```
**2️⃣ Install dependencies**
```
pip install -r requirements.txt
```
**3️⃣ Create a Groq API key of your own**
```
https://console.groq.com/keys
```
**4️⃣ Run the chatbot**
```
python app.py
```

## **🔧 How It Works**
1. **Upload a PDF or DOCX document**  
   - The chatbot allows users to upload documents for knowledge retrieval.

2. **Document Preprocessing**  
   - The document is **loaded and chunked** using `PyPDFLoader` (for PDFs) or `python-docx` (for DOCX).  
   - Text is **split into smaller chunks** using `RecursiveCharacterTextSplitter`.  
   - Each chunk is **preprocessed** (lowercased, cleaned, stemmed using `Sastrawi`).

3. **Embedding & Indexing**  
   - The chunks are **embedded** using `SentenceTransformer`.  
   - The embeddings are stored in a **FAISS vector database** for **fast similarity search**.  
   - The raw text chunks are also indexed using `BM25Okapi` for **keyword-based retrieval**.

4. **Retrieval-Augmented Generation (RAG) Process**  
   - When a user asks a question, the chatbot:  
     🔹 **Finds the most relevant chunks** using FAISS (vector search).  
     🔹 **Performs keyword search** using BM25 (for better relevance).  
     🔹 **Combines both retrieval results** for improved accuracy.  

5. **Response Generation**  
   - The retrieved context is passed to **LangChain** with `ChatGroq` (Groq API).  
   - The LLM generates a response based on the retrieved knowledge.  
   - The chatbot displays the answer in **Gradio UI**.

## **📦 Dependencies**
To install all required dependencies, create a `requirements.txt` with:
```txt
gradio
numpy
sqlite3
faiss-cpu
sentence-transformers
langchain
langchain_groq
langchain_community
rank-bm25
python-dotenv
sastrawi
pandas
python-docx
matplotlib
faker
```
Then, install using:
```
pip install -r requirements.txt
```

## **📖 References**
-----------------

- **FAISS**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **BM25 (rank_bm25)**: [https://github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)
- **LangChain**: [https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain)
- **Gradio**: [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)
- **Groq API**: [https://groq.com](https://groq.com)