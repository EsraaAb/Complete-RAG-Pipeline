from Ingestion_pipeline import embedder 
import chromadb 
import requests 




def query_chromadb(question, top_k=3, collection_name="rag_chunks"):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)

    question_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings = [question_embedding],
        n_results = top_k
    )

    

    chunks = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            chunks.append({
                "text": doc,
                "filename": results['metadatas'][0][i]['filename'],
                "chunk_index": results['metadatas'][0][i]['chunk_index'],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
    
    return chunks






def build_prompt(question, chunks):
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Source {i+1}: {chunk['filename']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Get unique source filenames for citation
    sources = list(set([chunk['filename'] for chunk in chunks]))
    sources_text = ", ".join(sources)
    
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
    
    return prompt, sources_text






def ask_ollama(prompt, model="phi3"):
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






def ask_question(question, top_k=3, model="phi3"):
    print(f"\n Question: {question}")
    
    chunks = query_chromadb(question, top_k=top_k)
    
    if not chunks:
        return "No relevant information found in the documents.", []
    

    confidence = calculate_confidence(chunks)
    print(f"📊 Confidence: {confidence['level']} ({confidence['score']})")
    
    prompt, sources = build_prompt(question, chunks)
    
    answer = ask_ollama(prompt, model=model)
    
    return answer, list(set([chunk['filename'] for chunk in chunks])), confidence 





def calculate_confidence(chunks, threshold_high=0.5, threshold_low=0.8):
    if not chunks:
        return {"score": 0, "level": "low", "icon": "❌", "message": "No relevant chunks found"}
    
    # Calculate average distance of top chunks
    distances = [chunk.get('distance', 1.0) for chunk in chunks if chunk.get('distance')]
    if not distances:
        return {"score": 0, "level": "low", "icon": "❌", "message": "No distance data available"}
    
    avg_distance = sum(distances) / len(distances)
    
    # Convert to confidence (0-1 scale)
    confidence_score = max(0, min(1, 1 - avg_distance))
    
    # Determine level
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