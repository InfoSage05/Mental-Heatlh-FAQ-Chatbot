import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Preprocess text by lowercasing, removing punctuation/stopwords, and tokenizing."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = word_tokenize(text)         #Using nltk tokenizer for this purpose.
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
    return ' '.join(tokens)

def process_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Loading, cleaning, and saving FAQ data."""
    df = pd.read_csv("E:\Machine Learning\RAG\Mental_Health_Chatbot\data\raw\Mental_Health_FAQ.csv")
    
    # Cleaning questions and answers
    df['clean_question'] = df['question'].apply(clean_text)
    df['clean_answer'] = df['answer'].apply(clean_text)
    
    # Saving the processed data in the data/processed directory
    df.to_csv("E:\Machine Learning\RAG\Mental_Health_Chatbot\data\processed\cleaned_faq.csv", index=False)
    return df

if __name__ == "__main__":
    input_csv = "E:\Machine Learning\RAG\Mental_Health_Chatbot\data\raw\Mental_Health_FAQ.csv"
    output_csv = "E:\Machine Learning\RAG\Mental_Health_Chatbot\data\processed\cleaned_faq.csv"
    process_dataset(input_csv, output_csv)
    print("Data preprocessing complete!")