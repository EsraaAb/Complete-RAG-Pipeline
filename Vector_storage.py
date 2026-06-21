import chromadb 
from shared import embedder 

def store_chromadb(chunks_with_metadata, collection_name = "pre_calculated_chunks"):
    client = chromadb.PersistentClient(path="./chroma_db")


    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass 


    collection = client.create_collection(collection_name)
    print(f"Created new collection: {collection_name}")



    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks_with_metadata):
        ids.append(f"chunk_{i}")
        
        # Generate embedding for each chunk 
        embedding = embedder.encode(chunk["text"]).tolist()
        embeddings.append(embedding)
        
        documents.append(chunk["text"])
        metadatas.append({
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"]
        })
    
   
    # processing 100 chunks at a time. 
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Stored chunks {i} to {end}")
    
    print(f"Stored {len(ids)} chunks in collection '{collection_name}'")
    return len(ids)
    