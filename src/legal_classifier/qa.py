import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

UNAVAILABLE = "Answer not in context."
_MODEL = None
_GENERATOR = None
_EMBEDDER = None
_TOKENIZER = None
_QA_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _chunk_text(text: str, max_words: int = 140, overlap_words: int = 30) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [_normalize(text)]

    chunks: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
            continue

        step = max(1, max_words - overlap_words)
        for start in range(0, len(words), step):
            piece = words[start : start + max_words]
            if not piece:
                continue
            chunks.append(" ".join(piece))
            if start + max_words >= len(words):
                break

    return [c for c in chunks if c]


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDER
    except Exception:
        return None


def _semantic_retrieve(chunks: List[str], question: str, top_k: int) -> List[Tuple[int, float]]:
    embedder = _get_embedder()
    if embedder is None:
        return []

    chunk_emb = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = chunk_emb @ q_emb
    order = np.argsort(scores)[::-1][: max(1, min(top_k, len(chunks)))]
    return [(int(i), float(scores[int(i)])) for i in order]


def _tfidf_retrieve(chunks: List[str], question: str, top_k: int) -> List[Tuple[int, float]]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks + [question])
    chunk_vecs = matrix[:-1]
    q_vec = matrix[-1]
    scores = (chunk_vecs @ q_vec.T).toarray().ravel()
    order = np.argsort(scores)[::-1][: max(1, min(top_k, len(chunks)))]
    return [(int(i), float(scores[int(i)])) for i in order]


def _retrieve(chunks: List[str], question: str, top_k: int):
    ranked = _semantic_retrieve(chunks, question, top_k)
    method = "semantic"
    if not ranked:
        ranked = _tfidf_retrieve(chunks, question, top_k)
        method = "tfidf"

    evidence = [{"snippet": chunks[i], "score": s} for i, s in ranked]
    return evidence, method


def _get_generator():
    global _MODEL, _TOKENIZER

    if _MODEL and _TOKENIZER:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(_QA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        _QA_MODEL_ID,
        device_map="auto",
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
    )

    _MODEL = model
    _TOKENIZER = tokenizer
    return model, tokenizer


def _build_messages(question: str, evidence: List[Dict]):
    context = "\n\n".join(
        f"{item['snippet']}" for item in evidence[:3]
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a legal reasoning assistant. "
                "Answer in three short paragraphs: "
                "1) identify the legal issue, "
                "2) apply the governing legal principle to the facts, "
                "3) provide a clear conclusion. "
                "Do not copy the context text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            ),
        },
    ]


def answer_question(document_text: str, question: str, top_k: int = 3) -> Dict:
    doc = _normalize(document_text)
    q = _normalize(question)

    if not doc:
        return {"answer": "No document text available.", "evidence": []}
    if not q:
        return {"answer": "Please enter a question.", "evidence": []}

    chunks = _chunk_text(doc)
    evidence, method = _retrieve(chunks, q, top_k)

    if not evidence:
        return {"answer": UNAVAILABLE, "evidence": evidence}

    model, tokenizer = _get_generator()
    messages = _build_messages(q, evidence)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.0,  # deterministic
        do_sample=False,  # no sampling
        repetition_penalty=1.2,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt prefix if present
    if prompt in decoded:
        decoded = decoded.replace(prompt, "").strip()

    return {"answer": decoded.strip(), "evidence": evidence}
