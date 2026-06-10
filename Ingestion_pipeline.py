import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path 
import nltk 
from nltk.tokenize import sent_tokenize 
nltk.download('punkt_tab')

from sentence_transformers import SentenceTransformer
## loading the embedding model that will be used for sentence embedding so they can be compared using cosine similarity. 
embedder = SentenceTransformer('all-MiniLM-L6-v2')

from Vector_storage import store_chromadb 


## loading the files from the folder and append it to a list. 
def loading_files_from_folder(folder_path):
    document_content = []
    folder_path = Path(folder_path)
    for file in folder_path.glob("*.txt"):
        with file.open("r", encoding="utf-8") as fh:
            content = fh.read()
            document_content.append({
                "filename": file.name,
                "content": content
            }) 
            print(f"loading {file.name}")
    return document_content 

documents = loading_files_from_folder(folder_path = "Data")




## splitting the document into sentences. 
def splitting_document_into_sentences(document):
    splitting = sent_tokenize(document)
    return splitting 



# converting each sentences to its vector embedding so it can be used for similarity compare later 
def sentence_embeddings(sentences, threshold = 0.7):
    embeddings = []
    for sentence in sentences:
        embedding = embedder.encode(sentence) 
        embeddings.append(embedding) 
    return embeddings


# the formula to calculate the similarity between each consecutive sentences to calculate the similarity score between them.  
def cosine_similarity(sentence1, sentence2):
    dot_product = sum(a * b for a, b in zip(sentence1, sentence2))
    norm_v1 = sum(a * a for a in sentence1) ** 0.5
    norm_v2 = sum(b * b for b in sentence2) ** 0.5
    return dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) else 0




# for each document, it checks if the similarity score between each consecutive sentences is larger or equal to the threshold to group the chunks. 
def group_sentences_into_chunks(sentences, similarities ,threshold = 0.7):
    chunks =[]
    current_chunk = [sentences[0]]
    
    
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            current_chunk.append(sentences[i + 1])

        else:
            chunks.append(current_chunk)
            current_chunk = [sentences[i + 1]]

    chunks.append(current_chunk)

    return chunks





all_chunks_with_metadata = []

for doc in documents:
    similarities = []


    # calling the function to split the document into sentences 
    sentences = splitting_document_into_sentences(doc["content"])


    # calling the function to create the embeddings for each sentence
    embeddings = sentence_embeddings(sentences)

    # looping through those embeddings to calculate the similarties scores 
    for i in range(len(embeddings)-1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)

    
    # creating the chunks 
    chunks = group_sentences_into_chunks(sentences, similarities, threshold=0.7)

    print(f"\n{doc['filename']}:")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Chunks: {len(chunks)}")
    
    
    for chunk_idx, chunk_sentences in enumerate(chunks):
        all_chunks_with_metadata.append({
            "text": " ".join(chunk_sentences),
            "filename": doc["filename"],
            "chunk_index": chunk_idx,
            "sentence_count": len(chunk_sentences)
        })




# calling the function to store the chunks in the chroma database. 
store_chromadb(all_chunks_with_metadata, collection_name="rag_chunks")


