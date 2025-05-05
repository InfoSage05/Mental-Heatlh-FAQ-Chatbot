import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from src.generation import AnswerGenerator
from src.retrieval import setup_pinecone_index
from src.utils import Evaluator
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv() # Loading .env variables

# Initializing components
@st.cache_resource
def load_components():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = AnswerGenerator()
    evaluator = Evaluator()
    index = setup_pinecone_index("mental-health-faq")
    faq_data = pd.read_csv("data/processed/cleaned_faq.csv")
    return model, generator, evaluator, index, faq_data

model, generator, evaluator, index, faq_data = load_components()

# Streamlit UI
st.title("Mental Health FAQ Chatbot")
user_query = st.text_input("Ask your question:")

if user_query:
    # Retrieving context
    query_embedding = model.encode(user_query)
    results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True)
    context = [match.metadata["answer"] for match in results.matches]
    
    # Generating the answer
    answer = generator.generate_answer(context, user_query)
    
    # Find best FAQ match for evaluation
    best_match_id = results.matches[0].id
    reference_answer = faq_data.iloc[int(best_match_id)]["answer"]
    
    # Evaluation metrics
    bleu_score = evaluator.compute_bleu(answer, reference_answer)
    rouge_score = evaluator.compute_rouge(answer, reference_answer)
    
    # Displaying results
    st.subheader("Generated Answer")
    st.write(answer)
    
    st.subheader("Retrieved Context")
    for i, ans in enumerate(context, 1):
        st.write(f"**FAQ {i}:** {ans}")
    
    st.subheader("Evaluation Metrics")
    st.write(f"BLEU-4 Score: {bleu_score:.4f}")
    st.write(f"ROUGE-L F1: {rouge_score['rougeL'].fmeasure:.4f}")

