## Handles Data Processing & Retrieval
import pandas as pd
import string
import faiss
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Create embeddings and store in FAISS index
def create_faiss_index(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['question_clean'].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

# Retrieve top-k similar FAQs
def retrieve_faq(query, df, model, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]






