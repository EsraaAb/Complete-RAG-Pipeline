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



## Installation

### 1. Clone the repository
git clone https://github.com/EsraaAb/Complete-RAG-Pipeline.git
cd complete-rag-pipeline



### 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Install dependencies
pip install -r requirements.txt



### 4. Install Ollama
#### Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

#### Windows: Download from https://ollama.com/download


### 5. Pull a model
ollama pull phi3




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

