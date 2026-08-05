import math
import re
from .document_types import DOC_TYPE_KEYWORDS, DOC_TYPE_PROTOTYPES

# ---- Embedding backend ----
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _USE_ST = True
except Exception:
    _model = None
    _USE_ST = False


def _tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def embed(texts):
    """Return list of vectors (real embeddings if ST installed, else fallback)."""
    if _USE_ST:
        return [list(v) for v in _model.encode(texts, show_progress_bar=False)]
    dim = 256
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for tok in _tokenize(t):
            v[hash(tok) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# pre-compute prototype embeddings once (at import)
_TYPE_NAMES = list(DOC_TYPE_PROTOTYPES.keys())
_TYPE_VECS = embed([DOC_TYPE_PROTOTYPES[t] for t in _TYPE_NAMES])

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


def _keyword_score(text_lower, doc_type):
    kws = DOC_TYPE_KEYWORDS.get(doc_type, [])
    if not kws:
        return 0.0
    hits = sum(1 for kw in kws if kw in text_lower)
    return hits / len(kws)


def classify(text, precomputed_vec=None):
    """
    Hybrid classification.
    precomputed_vec: OPTIONAL - pass an already-computed document embedding
                     to AVOID embedding the document twice (see RAG note).
    Returns: {"document_type", "confidence", "method"}
    """
    if not text or not text.strip():
        return {"document_type": "Unknown", "confidence": 0.0, "method": "hybrid"}

    text_lower = text.lower()

    # reuse the embedding if provided, else compute from first 2000 chars
    doc_vec = precomputed_vec if precomputed_vec is not None else embed([text[:2000]])[0]

    best, best_score = "Other", 0.0
    for name, tvec in zip(_TYPE_NAMES, _TYPE_VECS):
        sem = max(0.0, _cosine(doc_vec, tvec))
        kw = _keyword_score(text_lower, name)
        final = SEMANTIC_WEIGHT * sem + KEYWORD_WEIGHT * kw
        if final > best_score:
            best, best_score = name, final

    if best_score < 0.15:
        return {"document_type": "Other", "confidence": round(float(best_score), 2), "method": "hybrid"}

    confidence = round(float(min(best_score * 1.3, 0.99)), 2)
    method = "semantic+keyword" + (" (transformer)" if _USE_ST else " (fallback)")
    return {"document_type": best, "confidence": confidence, "method": method}
 