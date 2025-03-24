import os
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
faiss_index_path = "mikovexa_faiss_index.bin"
if os.path.exists(faiss_index_path):
    print("Loading existing FAISS index...")
    faiss_index = faiss.read_index(faiss_index_path)
else:
    print("No FAISS index found. You need to generate it first.")

def load_bm25_index(texts):
    tokenized_texts = [text.split() if isinstance(text, str) else text for text in texts]
    return BM25Okapi(tokenized_texts)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
