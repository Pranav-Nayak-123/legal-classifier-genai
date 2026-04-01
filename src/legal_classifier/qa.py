import re
from typing import Dict, List, Tuple
from threading import Thread

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

UNAVAILABLE = "Answer not in context."
_MODEL = None
_GENERATOR = None
_EMBEDDER = None
_TOKENIZER = None
_QA_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ROLE_INSTRUCTIONS = {
    "law_student": "Explain clearly and educationally with legal accuracy.",
    "paralegal": "Be practical, concise, and checklist-oriented.",
    "counsel": "Provide professional legal analysis with risk awareness.",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _chunk_text(text: str, max_words: int = 100, overlap_words: int = 20) -> List[str]:
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


def _build_messages(
    question: str,
    evidence: List[Dict],
    role_template: str = "counsel",
    history: List[Dict] | None = None,
):
    context = "\n\n".join(
        f"{item['snippet']}" for item in evidence[:2]
    )
    role_hint = ROLE_INSTRUCTIONS.get(role_template, ROLE_INSTRUCTIONS["counsel"])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal reasoning assistant. "
                f"{role_hint} "
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

    # Add short rolling memory from prior turns.
    if history:
        history_tail = history[-4:]
        memory_blocks = []
        for turn in history_tail:
            q_prev = str(turn.get("question", "")).strip()
            a_prev = str(turn.get("answer", "")).strip()
            if q_prev and a_prev:
                memory_blocks.append(f"Q: {q_prev}\nA: {a_prev}")
        if memory_blocks:
            messages[1]["content"] += "\n\nRecent conversation:\n" + "\n\n".join(memory_blocks)

    return messages


def answer_question(
    document_text: str,
    question: str,
    top_k: int = 2,
    role_template: str = "counsel",
    history: List[Dict] | None = None,
) -> Dict:
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
    messages = _build_messages(q, evidence, role_template=role_template, history=history)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.0,  # deterministic
        do_sample=False,  # no sampling
        repetition_penalty=1.2,
        use_cache=True,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt prefix if present
    if prompt in decoded:
        decoded = decoded.replace(prompt, "").strip()

    return {"answer": decoded.strip(), "evidence": evidence}


def retrieve_evidence(document_text: str, question: str, top_k: int = 2) -> List[Dict]:
    doc = _normalize(document_text)
    q = _normalize(question)
    if not doc or not q:
        return []
    chunks = _chunk_text(doc)
    evidence, _method = _retrieve(chunks, q, top_k)
    return evidence


def stream_answer_question(
    document_text: str,
    question: str,
    top_k: int = 2,
    role_template: str = "counsel",
    precomputed_evidence: List[Dict] | None = None,
    history: List[Dict] | None = None,
):
    doc = _normalize(document_text)
    q = _normalize(question)

    if not doc:
        yield "No document text available."
        return
    if not q:
        yield "Please enter a question."
        return

    evidence = precomputed_evidence or retrieve_evidence(doc, q, top_k=top_k)
    if not evidence:
        yield UNAVAILABLE
        return

    model, tokenizer = _get_generator()
    if model is None or tokenizer is None:
        yield "Model is not ready."
        return

    messages = _build_messages(q, evidence, role_template=role_template, history=history)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(model.device)

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": 120,
        "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "use_cache": True,
        "streamer": streamer,
    }

    worker = Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    for text in streamer:
        if text:
            yield text
    worker.join()
