# Document-Based Research & RAG System

A modular toolkit designed for processing, indexing, and analyzing large-scale document archives. This system specializes in transforming unstructured or semi-structured data (HTM, PDF, OCR) into a high-performance Retrieval-Augmented Generation (RAG) pipeline.

## Core Concepts

The system operates on a **Page-Granular Architecture**, ensuring that retrieved context is pinned to specific physical or logical pages. This makes it ideal for research where citation and source verification are critical.

### 1. Data Ingestion & Structuring
The system ingests documents from multiple sources and normalizes them into a unified JSON schema:
*   **Shamela HTM Parsing:** Extracts text from legacy HTML exports, preserving page markers and volume structures.
*   **PDF OCR & Extraction:** Utilizes `PyMuPDF` and OCR layers to convert scanned documents into searchable text while maintaining page-level mapping.
*   **Schema:** `book_name -> { page_number: "text_content" }`

### 2. Vector Indexing
To enable semantic search, the system implements a dense vector retrieval layer:
*   **Embedding Model:** Uses state-of-the-art multilingual models (e.g., BGE-M3) to generate high-dimensional embeddings.
*   **Storage:** Indices and metadata are persisted locally (`.npz`, `.json`), allowing for fast similarity searches without requiring an external vector database.

### 3. Retrieval-Augmented Generation (RAG)
The RAG pipeline bridges the gap between static archives and Large Language Models (LLMs):
*   **Retrieval:** Performs semantic lookups across the indexed volumes to find the most relevant pages for a given query.
*   **Context Injection:** Feeds the retrieved text directly into the model prompt.
*   **Inference:** Optimized for models like Qwen to provide grounded, accurate answers based strictly on the provided corpus.

### 4. Interactive Interface
The system includes a Gradio-based web interface (or CLI) for real-time querying and document exploration.

## Technical Stack

*   **Logic:** Python 3.11+
*   **ML Framework:** PyTorch (v2.6.0+ with CUDA support)
*   **Embeddings:** HuggingFace Transformers & FlagEmbedding (BGE)
*   **Document Processing:** PyMuPDF (fitz)
*   **Interface:** Gradio

## Usage Flow

1.  **Prepare:** Place HTM files in `books/htm-files` or PDFs in `books/pdfs`.
2.  **Process:** Run the ingestion scripts to populate `books.json`.
3.  **Index:** Generate the vector embeddings for the text corpus.
4.  **Query:** Launch the QA script to interact with the documents via semantic search.
