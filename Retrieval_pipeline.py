from shared import embedder 
from BM25_retriever import BM25Retriever
import chromadb 
import requests 


class RAG_Pipeline(): 
    def __init__(self, collection_name="rag_chunks"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.bm25_retriever = BM25Retriever()
        self.all_chunks = [] 
    
    
    def load_chunks_for_bm25(self, chunks):
        """Store chunks and build BM25 index"""
        self.all_chunks = chunks
        self.bm25_retriever.build_index(chunks)
        print(f"BM25 ready with {len(chunks)} chunks")
    
   


    def vector_search(self, question, top_k=10):
        """Vector search returning results with ranks"""
        collection = self.client.get_collection(name=self.collection_name)
        question_embedding = embedder.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                chunks.append({
                    'text': doc,
                    'filename': results['metadatas'][0][i]['filename'],
                    'chunk_index': results['metadatas'][0][i]['chunk_index'],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'vector_rank': i + 1
                })
        return chunks
    
   

    def bm25_search(self, question, top_k=10):
        """BM25 search"""
        return self.bm25_retriever.search(question, top_k=top_k)
   



    def hybrid_search(self, question, top_k=10, rrf_k=60):
        """Hybrid search combining vector and BM25 with RRF"""

        vector_results = self.vector_search(question, top_k=20)
        print(f"🔍 Vector: {len(vector_results)} results")
        
        bm25_results = self.bm25_search(question, top_k=20)
        print(f"🔍 BM25: {len(bm25_results)} results")
        
        combined = {}
        
        # Add vector results
        for rank, result in enumerate(vector_results):
            rrf_score = 1 / (rrf_k + rank + 1)
            key = result['text'][:100]
            combined[key] = {
                'text': result['text'],
                'filename': result['filename'],
                'chunk_index': result['chunk_index'],
                'rrf_score': rrf_score,
                'vector_rank': rank + 1,
                'bm25_rank': None,
                'distance': result.get('distance')
            }
        
        for rank, result in enumerate(bm25_results):
            rrf_score = 1 / (rrf_k + rank + 1)
            key = result['text'][:100]
            
            if key in combined:
                combined[key]['rrf_score'] += rrf_score
                combined[key]['bm25_rank'] = rank + 1
            else:
                combined[key] = {
                    'text': result['text'],
                    'filename': result['filename'],
                    'chunk_index': result['chunk_index'],
                    'rrf_score': rrf_score,
                    'vector_rank': None,
                    'bm25_rank': rank + 1,
                    'distance': None
                }
        
        results = list(combined.values())
        results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        print(f"📊 Hybrid: {len(results)} combined results")
        return results[:top_k]




    def build_prompt(self, question, chunks):
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}: {chunk['filename']}]\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        sources = list(set([chunk['filename'] for chunk in chunks]))
        
        prompt = f"""You are a helpful assistant answering questions based ONLY on the provided documents.

DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the documents above
2. If the answer is not in the documents, say "I cannot answer based on the provided documents"
3. Cite your sources using [Source X] format
4. Be concise and accurate

ANSWER:"""
        
        return prompt, sources






    def ask_ollama(self, prompt, model="phi3"):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Ollama error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    




    def ask_question(self, question, top_k=3, model="phi3"):
        print(f"\n📝 Question: {question}")
        
        # Use hybrid search instead of just vector
        chunks = self.hybrid_search(question, top_k=top_k)
        
        if not chunks:
            return "No relevant information found in the documents.", [], None
        
        confidence = self.calculate_confidence(chunks)
        print(f"📊 Confidence: {confidence['level']} ({confidence['score']})")
        
        prompt, sources = self.build_prompt(question, chunks)
        
        answer = self.ask_ollama(prompt, model=model)
        
        return answer, list(set([chunk['filename'] for chunk in chunks])), confidence

  
  
  
  
  
    def calculate_confidence(self, chunks, threshold_high=0.5, threshold_low=0.8):
        if not chunks:
            return {"score": 0, "level": "low", "icon": "❌", "message": "No relevant chunks found"}
        
        distances = [chunk.get('distance', 1.0) for chunk in chunks if chunk.get('distance')]
        if not distances:
            return {"score": 0, "level": "low", "icon": "❌", "message": "No distance data available"}
        
        avg_distance = sum(distances) / len(distances)
        confidence_score = max(0, min(1, 1 - avg_distance))
        
        if avg_distance < threshold_high:
            level = "high"
            icon = "✅"
            message = "High confidence - Answer is well-supported by documents"
        elif avg_distance < threshold_low:
            level = "medium"
            icon = "⚠️"
            message = "Medium confidence - Answer has moderate support"
        else:
            level = "low"
            icon = "❌"
            message = "Low confidence - Answer may not be fully supported by documents"
        
        return {
            "score": round(confidence_score, 3),
            "distance": round(avg_distance, 3),
            "level": level,
            "icon": icon,
            "message": message
        }