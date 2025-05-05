import os
import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Loading .env variables

# Initializing Pinecone as Environment Variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

# Initializing embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def setup_pinecone_index(index_name: str = "mental-health-faq") -> pinecone.Index:
    """Create or connect to a Pinecone index. It means that we are connecting our pipeline to the cloud database 
    where we will store the embeddings of the FAQs with the help of the Pinecone API."""
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 uses 384-dimensional embeddings
            metric="cosine"
        )
    return pinecone.Index(index_name)

def upload_embeddings_to_pinecone(df: pd.DataFrame, index: pinecone.Index, batch_size: int = 32) -> None:
    """Convert FAQs to embeddings and upload to Pinecone."""
    questions = df['clean_question'].tolist()
    answers = df['clean_answer'].tolist()
    
    # Generating embeddings
    embeddings = model.encode(questions, show_progress_bar=True)
    
    # Preparing data for Pinecone (id, vector, metadata)
    vectors = []
    for i, (emb, question, answer) in enumerate(zip(embeddings, questions, answers)):
        vectors.append({
            "id": str(i),
            "values": emb.tolist(),
            "metadata": {"question": question, "answer": answer}
        })
    
    # Batch upload
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    print(f"Uploaded {len(vectors)} vectors to Pinecone.")

if __name__ == "__main__":
    # Loading processed data
    df = pd.read_csv("E:\Machine Learning\RAG\Mental_Health_Chatbot\data\processed\cleaned_faq.csv")
    
    # Connecting to Pinecone Vector Database
    index = setup_pinecone_index()
    
    # Uploading embeddings to Vector Database
    upload_embeddings_to_pinecone(df, index)