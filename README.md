# Basic RAG Pipeline
A production-ready RAG (Retrieval-Augmented Generation) system with semantic chunking, vector search, confidence scoring, and source citations. Built with ChromaDB, Sentence Transformers, and Ollama.

## Features

-  **Multi-document ingestion** – Load and process multiple text files
-  **Semantic chunking** – Groups sentences based on similarity (not fixed size)
-  **Vector search** – ChromaDB for efficient similarity search
-  **Confidence scoring** – High/Medium/Low confidence for each answer
-  **Source citations** – Shows which documents answers came from
-  **Hallucination detection** – Flags low-confidence answers
-  **Web interface** – Gradio UI with examples
-  **100% local** – No API keys, no data leaves your computer



## Project Structure
complete-rag-pipeline/
├── Ingestion_pipeline.py # Load, chunk, embed, store
├── Vector_storage.py # ChromaDB operations
├── Retrieval_pipeline.py # Query, search, generate, confidence
├── app.py # Gradio web interface
├── Data/ # Place your .txt files here
├── chroma_db/ # Vector database (auto-generated)
└── requirements.txt



## Installation

### 1. Clone the repository
git clone https://github.com/yourusername/complete-rag-pipeline.git
cd complete-rag-pipeline



### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Install dependencies
pip install -r requirements.txt



### 4. Install Ollama
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download


### 5. Pull a model
ollama pull phi3




## Usage
### Step 1: Add your documents
Place .txt files in the Data/ folder:

text
Data/
├── ai.txt
├── ml.txt
└── nlp.txt



### Step 2: Run the ingestion pipeline
python Ingestion_pipeline.py

This will:

- Load all text files

- Split into sentences

- Generate embeddings

- Group sentences into semantic chunks

- Store in ChromaDB

### Step 3: Start Ollama (keep this terminal open)
ollama serve


### Step 4: Launch the web interface
python app.py


### Step 5: Open your browser
Go to http://127.0.0.1:7860




## Example
Question: What is machine learning?

Answer:
✅ Confidence: HIGH (Score: 0.82)
High confidence - Answer is well-supported by documents

Answer: Machine learning is a subset of AI that enables systems to 
learn from data without being explicitly programmed. It has three 
main types: supervised, unsupervised, and reinforcement learning.

Sources: ML.txt



## Technologies:
1) NLTK Library : Splitting the document into sentences.  
2) Sentence Transformers : Sentence embeddings. 
3) chromadb : vector database storage. 
4) Ollama & phi-3 : for LLM. 
5) Gradio : UI






## Phase 2: Add BM25
- Store all chunks in a list for BM25
- Tokenize chunks (split into words, lowercase)
- Build BM25 index using rank_bm25 library
- Implement BM25 search function
- Display BM25 results separately to compare


## Phase 3: Hybrid Search with RRF
- Get top 10 results from vector search (keep ranks)
- Get top 10 results from BM25 search (keep ranks)
- Implement RRF formula: score = 1/(k + rank) with k=60
- Merge both result sets into one ranked list
- Display hybrid results and compare with pure vector


## Phase 4: Reranking
- Load cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
- Take top 10-20 results from hybrid search
- For each result, compute (query, chunk) relevance score
- Re-sort by cross-encoder score
- Take top 3-5 for final answer