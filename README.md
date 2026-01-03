# SEC Filing Summarizer & Q&A (RAG) (F7) 

A retrieval-augmented generation (RAG) system that allows investors to query SEC 10-K and 10-Q filings and receive answers with source citations.

## Problem Statement

Investors need to quickly find relevant information from SEC filings (10-K annual reports, 10-Q quarterly reports) without manually reading thousands of pages. This project provides:
- Natural language search across SEC documents
- Accurate answers backed by source citations
- Quick access to specific sections and disclosures

## Dataset

The project uses the [SEC Filings dataset](https://www.kaggle.com/datasets/kharanshuvalangar/sec-filings) from Kaggle containing 10-K and 10-Q filings from various companies.

## Tech Stack

- **Language**: Python 3.8+
- **Document Processing**: Unstructured (for parsing PDFs and text)
- **RAG Framework**: LangChain
- **Embeddings**: OpenAI/Hugging Face embeddings
- **Vector Store**: FAISS or Pinecone
- **Data Parsing**: pypdf, pydantic

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd sec-filing-qa
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys (OpenAI, etc.)
   ```

5. Download SEC filings from [Kaggle](https://www.kaggle.com/datasets/kharanshuvalangar/sec-filings) and place in `data/filings/` folder

### Running the Project

**Index SEC filings into vector store:**
```bash
python index_filings.py
```

This will:
- Parse PDF and text SEC filings
- Split documents into chunks
- Generate embeddings for each chunk
- Store in vector database (FAISS/Pinecone)

**Ask questions about filings:**
```bash
python ask.py "What was the revenue growth in 2023?"
```

This will:
- Search the vector store for relevant chunks
- Generate an answer using LLM
- Return answer with source citations and URLs

**Start FastAPI server (optional):**
```bash
python -m uvicorn api:app --reload --port 8000
```

Then query via:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risks mentioned?"}'
```

## How It Works

### 1. Document Indexing

- SEC filings (10-K, 10-Q) are parsed and extracted
- Documents are split into chunks (typically 500-1000 tokens)
- Each chunk is converted to embeddings (numerical representations)
- Embeddings are stored in a vector database for fast similarity search

### 2. Question Answering

- User question is converted to an embedding
- Similar document chunks are retrieved from the vector store
- Retrieved chunks are passed to an LLM (e.g., GPT-4) as context
- LLM generates an answer citing specific sources and chunk locations

### 3. Source Attribution

- Each answer includes the specific filing name, section, and page number
- URLs to original filings are provided for verification
- Users can trace answers back to source documents


## Features

- Multi-filing support (can index multiple companies and years)
- Source citations with document references
- Configurable chunk size and overlap
- Support for multiple embedding models
- Optional caching for frequently asked questions


## Project Goals

- Build a searchable index of SEC filings
- Enable natural language queries across documents
- Provide answers with proper source attribution
- Deploy as API for investor tools

## Status

Initial setup and core infrastructure. Development in progress.

## Contact

For questions or contributions, please reach out to the team.
