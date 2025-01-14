# PDF RAG with Multi-Agent System

An intelligent document question-answering system that combines Retrieval-Augmented Generation (RAG) with a multi-agent architecture to provide accurate answers from PDF documents.

## Features

- **Multi-Agent Architecture**
  - Main Agent: Orchestrates the workflow between index and collection-based approaches
  - Index Agent: Handles PDF table of contents extraction and indexing
  - Collection Agent: Manages document embedding and vector storage
  - Answer Agent: Generates responses based on retrieved contexts

- **Dual Retrieval Approaches**
  - Index-based: Uses PDF table of contents for targeted page retrieval
  - Collection-based: Leverages vector embeddings for semantic search

- **Advanced Document Processing**
  - Automatic table of contents detection
  - Text chunking with overlap
  - BGE-M3 embeddings for semantic search
  - Hybrid ranking using ColBERT scores

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file with OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

3. Create storage directory:
```bash
mkdir chromadb_folder
```

## Usage

Run the interactive CLI:
```bash
python fully_agentic_flow-pdf-rag.py
```

The system will:
1. First attempt to use table of contents for targeted retrieval
2. Fall back to semantic search if needed
3. Generate answers based on the most relevant context

Type 'q' to exit the interactive session.

## Requirements

- Python 3.8+
- OpenAI API access
- ChromaDB
- PyMuPDF
- BGE-M3 embedding model
