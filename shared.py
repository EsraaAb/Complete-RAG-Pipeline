import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sentence_transformers import SentenceTransformer

# Load embedding model once, share everywhere
embedder = SentenceTransformer('all-MiniLM-L6-v2')