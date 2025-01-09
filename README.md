# PDF-RAG

A Retrieval-Augmented Generation system for querying PDF documents using advanced embedding and reranking techniques.

## Project Structure

The project consists of two main components:

### 1. Simple Chatbot (`chatbot.py`)
A straightforward implementation that:
- Accepts PDF documents
- Creates collections in the vector database
- Enables question-answering about submitted PDFs
- Checks for existing collections before creating new ones

### 2. Agentic Chatbot (`agentic/agentchat.py`)
An advanced implementation featuring:
- Natural language interaction
- Internal tool calling system with three main functions:
  - Collection creation
  - Existing collection checking
  - Answer retrieval
- Chat history maintenance during sessions
- Token limit monitoring with automatic session termination

## Technical Implementation

### PDF Preprocessing
- Utilizes `pymupdf4llm` library for PDF parsing
- Configuration:
  - Chunk size: 500 tokens
  - Overlap: 50 tokens
- Processes documents for ChromaDB ingestion

### Embedding and Retrieval
- **Embedding Model**: BGE (BAAI General Embedding)
- **Vector Database**: ChromaDB
- **Reranking System**: ColBERT scoring mechanism
  - Two-phase retrieval process:
    1. Initial document scoring using ColBERT query-document matching
    2. Contextual scoring with surrounding text

### Retrieval Logic
The retrieval system employs a sophisticated two-phase scoring approach:
1. Computes ColBERT scores for individual documents
2. Enhances scoring by considering surrounding context
3. Combines document-level and context-level scores for final ranking

For detailed implementation, refer to `get_full_context` function in `util.py`.

## Dependencies
- pymupdf4llm
- chromadb
- torch (for ColBERT scoring)
- transformers (for BGE embedding model)

## References
- Document Embedding: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- Vector Database: [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/introduction)
- PDF Parsing: [PyMuPDF Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)

## Usage

### Simple Chatbot
```python
python chatbot.py
# Follow prompts to:
# 1. Submit PDF path
# 2. Create collection
# 3. Query the document
```

### Agentic Chatbot
```python
python agentic/agentchat.py
# Enables:
# - Natural language interaction
# - Automatic collection management
# - Context-aware responses
```

## Performance Considerations
- Chat history is maintained until token limit (100,000) is reached
- Chunk size and overlap settings are optimized for typical document lengths
- Two-phase scoring ensures high-quality response relevance

## Contributing
Feel free to suggest improvements or report issues.
