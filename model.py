## model.py - LLM-based Response Generation

from transformers import pipeline

def load_llm():
    return pipeline("text-generation", model="meta-llama/Meta-Llama-3")

def generate_response(query, retrieved_faqs, llm):
    '''
    Generates a response to the user query using the retrieved FAQs and the LLM model.
    It is basically the prompt template that we are preparing for 
    the LLM model to generate the response.'''
    context = "\n".join(retrieved_faqs['answer'])
    prompt = f"Context: {context}\nUser Query: {query}\nResponse:"
    response = llm(prompt, max_length=200)
    return response[0]['generated_text']