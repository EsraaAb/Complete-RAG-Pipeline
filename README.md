# Basic RAG Pipeline
A production-ready RAG (Retrieval-Augmented Generation) system with **hybrid search** (vector + BM25), semantic chunking, confidence scoring, and source citations. Built with ChromaDB, Sentence Transformers, and Ollama.


## Features

- **Multi-document ingestion** – Load and process multiple text files
- **Semantic chunking** – Groups sentences based on similarity (not fixed size)
- **Hybrid Search** – Combines vector search + BM25 keyword search with RRF fusion
- **Vector search** – ChromaDB for efficient similarity search
- **BM25 keyword search** – Finds exact keyword matches for better precision
- **RRF Fusion** – Reciprocal Rank Fusion combines both search methods
- **Confidence scoring** – High/Medium/Low confidence for each answer
- **Source citations** – Shows which documents answers came from
- **Hallucination detection** – Flags low-confidence answers
- **Web interface** – Gradio UI with examples
- **100% local** – No API keys, no data leaves your computer

## 🆕 What's New (v2.0)

| Feature | Description |
|---------|-------------|
| **BM25 Retriever** | Keyword-based search to catch exact terms |
| **Hybrid Search** | Combines vector + BM25 using RRF |
| **RRF Fusion** | Reciprocal Rank Fusion for optimal results |



## Installation

### 1. Clone the repository
- git clone https://github.com/EsraaAb/Complete-RAG-Pipeline.git
- cd complete-rag-pipeline



### 2. Create virtual environment
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Install dependencies
- pip install -r requirements.txt



### 4. Install Ollama
#### Linux/Mac
- curl -fsSL https://ollama.com/install.sh | sh

#### Windows: Download from https://ollama.com/download


### 5. Pull a model
- ollama pull phi3




## Usage
### Step 1: Add your documents
Place .txt files in the Data/ folder:


### Step 2: Run the ingestion pipeline
python Ingestion_pipeline.py

This will:

- Load all text files

- Split into sentences

- Generate embeddings

- Group sentences into semantic chunks

- Store in ChromaDB

- Build BM25 index for hybrid search


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

## Demo 
<img width="1502" height="613" alt="Screenshot from 2026-06-10 04-56-20" src="https://github.com/user-attachments/assets/b479f6a5-88aa-4b55-9284-79ac4b8bd4d1" />


## Architecture
<img width="1407" height="768" alt="Gemini_Generated_Image_64ibjc64ibjc64ib" src="https://github.com/user-attachments/assets/75351787-d3a9-4c5f-93fd-08d553bc2a5c" />


## Technologies:
1) NLTK Library : Splitting the document into sentences.  
2) Sentence Transformers : Sentence embeddings. 
3) chromadb : vector database storage. 
4) Keyword Search	BM25 (rank_bm25)
5) Hybrid Search	Vector + BM25 with RRF
6) Ollama & phi-3 : for LLM. 
7) Gradio : UI

