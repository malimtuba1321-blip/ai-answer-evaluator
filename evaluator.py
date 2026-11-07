# evaluator.py
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from preprocessing import clean_text, tokenize


def compute_bleu(reference: str, candidate: str) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    return round(float(score), 4)


def compute_rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    ref_clean = clean_text(reference)
    cand_clean = clean_text(candidate)
    if not ref_clean or not cand_clean:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(ref_clean, cand_clean)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bertscore(reference: str, candidate: str) -> dict:
    if not reference.strip() or not candidate.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    P, R, F1 = bert_score_fn(
        [candidate], [reference],
        lang="en", model_type="bert-base-uncased", verbose=False
    )
    return {
        "precision": round(float(P[0]), 4),
        "recall": round(float(R[0]), 4),
        "f1": round(float(F1[0]), 4),
    }


def evaluate(reference: str, candidate: str) -> dict:
    bleu = compute_bleu(reference, candidate)
    rouge = compute_rouge(reference, candidate)
    bert = compute_bertscore(reference, candidate)
    composite = round((bleu + rouge["rougeL"] + bert["f1"]) / 3.0, 4)
    return {
        "BLEU": bleu,
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "ROUGE-L": rouge["rougeL"],
        "BERTScore-P": bert["precision"],
        "BERTScore-R": bert["recall"],
        "BERTScore-F1": bert["f1"],
        "Composite": composite,
    }