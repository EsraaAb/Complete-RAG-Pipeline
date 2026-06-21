from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict
import re 


class BM25Retriever():
    def __init__(self):
        self.chunks = []
        self.bm25 = None  




    def tokenize(self, text):
        words = text.lower()
        words = re.sub(r'[^\w\s]', '', words)
        list_of_words = words.split()
        return list_of_words




    def build_index(self, chunks):
        self.chunks = chunks
        
        tokenized_chunks = []
        for chunk in chunks:
            tokens = self.tokenize(chunk["text"])
            tokenized_chunks.append(tokens)

        self.bm25 = BM25Okapi(tokenized_chunks)
        print(f"BM25 index built with {len(chunks)} chunks")





    def search(self, query, top_k=5):
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                'text': chunk['text'], 
                'filename': chunk['filename'],
                'chunk_index': chunk['chunk_index'],
                'bm25_score': scores[idx]
            })
        return results