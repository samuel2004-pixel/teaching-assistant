import os
import math
import pickle
from pypdf import PdfReader
from collections import Counter

BOOKS_DIR = "data/books"
VECTOR_DIR = "data/vector_index"
os.makedirs(VECTOR_DIR, exist_ok=True)

documents = []
metadata = []

print("📂 Books:", os.listdir(BOOKS_DIR))

for file in os.listdir(BOOKS_DIR):
    if not file.lower().endswith(".pdf"):
        continue

    print("📖 Reading:", file)
    reader = PdfReader(os.path.join(BOOKS_DIR, file))
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "

    if not text.strip():
        print("⚠️ No text in", file)
        continue

    words = text.lower().split()
    for i in range(0, len(words), 200):
        chunk = words[i:i+200]
        documents.append(chunk)
        metadata.append({
            "source": file,
            "text": " ".join(chunk)
        })

print("✂️ Chunks:", len(documents))

# ---------- TF ----------
term_freqs = [Counter(doc) for doc in documents]

# ---------- IDF ----------
doc_count = len(documents)
idf = {}
for tf in term_freqs:
    for term in tf:
        idf[term] = idf.get(term, 0) + 1

for term in idf:
    idf[term] = math.log(doc_count / (1 + idf[term]))

store = {
    "tf": term_freqs,
    "idf": idf,
    "metadata": metadata
}

with open(os.path.join(VECTOR_DIR, "vector_store.pkl"), "wb") as f:
    pickle.dump(store, f)

print("🎉 TRAINING COMPLETE (PURE PYTHON)")
print("📦 Saved: data/vector_index/vector_store.pkl")
