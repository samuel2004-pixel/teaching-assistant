import pickle
import math
from collections import Counter

# Load knowledge store
store = pickle.load(open("data/vector_index/vector_store.pkl", "rb"))
tf_docs = store["tf"]
idf = store["idf"]
metadata = store["metadata"]

# ---------- Math utils ----------
def cosine(v1, v2):
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1)
    mag1 = math.sqrt(sum(x*x for x in v1.values()))
    mag2 = math.sqrt(sum(x*x for x in v2.values()))
    return dot / (mag1 * mag2 + 1e-9)

def vectorize(words):
    tf = Counter(words)
    return {w: tf[w] * idf.get(w, 0) for w in tf}

# ---------- Intent detection ----------
def detect_intent(query):
    q = query.lower()
    if q.startswith("what is") or "define" in q:
        return "definition"
    if "step" in q or "how to" in q:
        return "steps"
    if "difference" in q or "vs" in q:
        return "comparison"
    return "explanation"

# ---------- Advanced retrieval ----------
def retrieve_context(query, k=5):
    query_words = query.lower().split()
    q_vec = vectorize(query_words)
    intent = detect_intent(query)

    sentence_scores = []

    for i, doc_tf in enumerate(tf_docs):
        text = metadata[i]["text"]
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

        for s in sentences:
            words = s.lower().split()
            s_vec = vectorize(words)

            # TF-IDF similarity
            sim = cosine(q_vec, s_vec)

            # Keyword overlap boost
            overlap = len(set(query_words) & set(words))
            boost = overlap / (len(query_words) + 1)

            score = sim + boost

            if score > 0:
                sentence_scores.append((score, s))

    if not sentence_scores:
        return "No relevant explanation found in the study material."

    # Rank sentences
    top = sorted(sentence_scores, reverse=True)[:k]

    # ---------- Teaching formatting ----------
    if intent == "definition":
        return top[0][1]

    if intent == "steps":
        steps = []
        for i, (_, s) in enumerate(top, 1):
            steps.append(f"Step {i}: {s}")
        return "\n".join(steps)

    # Explanation (default)
    explanation = []
    for _, s in top:
        explanation.append(s)

    return ". ".join(explanation)
