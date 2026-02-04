import faiss
import pickle
import os

VECTOR_DIR = "data/vector_index"
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = f"{VECTOR_DIR}/index.faiss"
META_PATH = f"{VECTOR_DIR}/meta.pkl"

def save(index, meta):
    faiss.write_index(index, INDEX_PATH)
    pickle.dump(meta, open(META_PATH, "wb"))

def load():
    index = faiss.read_index(INDEX_PATH)
    meta = pickle.load(open(META_PATH, "rb"))
    return index, meta
