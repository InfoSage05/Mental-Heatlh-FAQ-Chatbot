## Evaluating Chatbot Responses
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from typing import List, Dict

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def compute_bleu(self, generated: str, reference: str) -> float:
        ref_tokens = reference.split()
        gen_tokens = generated.split()
        return sentence_bleu([ref_tokens], gen_tokens)
    
    def compute_rouge(self, generated: str, reference: str) -> Dict:
        return self.scorer.score(reference, generated)
    
    
## Got two different methods for the evaluation of the chatbot responses.    
def evaluate(df, llm):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for _, row in df.iterrows():
        reference = row['answer']
        generated = generate_response(row['question'], retrieve_faq(row['question'], df, model, index), llm)
        score = scorer.score(reference, generated)['rougeL'].fmeasure
        scores.append(score)
    return sum(scores) / len(scores)

print("Average ROUGE Score:", evaluate(df, llm))