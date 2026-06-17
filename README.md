<div align="center">
   <img width="705" height="142" alt="ascii-art-text (1)" src="https://github.com/user-attachments/assets/b153df10-333b-497f-ba93-b0b97a90fddf" />

   <p align="center">
  📄 CLI PDF Q&A • 🔍 RAG • 🗄️ ChromaDB • 🧠 Sentence Transformers • 🤖 OpenRouter
</p>
</div>

## Overview

Talk-to-PDF is a CLI-based question answering application that allows you to query documents locally. This project was built to learn the fundamentals of Retrieval-Augmented Generation (RAG) by combining document loading, text splitting, embeddings, vector databases, and LLM orchestration. It is meant purely as a personal educational tool and experimentation playground.

## Features

- **Local PDF Loading**: Scans a local data folder and loads PDF text page-by-page.
- **Recursive Character Splitting**: Chunks text into smaller segments to capture relevant context without overflow.
- **Local Embedding Generation**: Generates vector representations using Hugging Face's `sentence-transformers` library.
- **Vector Search Database**: Indexes documents in a local ChromaDB instance and retrieves similar chunks using cosine similarity.
- **State-Tracked Rebuilding**: Compares files against a cache state to detect additions, deletions, or modifications, rebuilding the vector store only when directory content changes.
- **LLM Completion**: Queries OpenRouter APIs to generate responses based on retrieved context.
- **Source-Aware Responses**: Output contains the answer alongside retrieved source file names, page numbers, similarity scores, and context.

## Tech Stack

- **Python**
- **uv** (for fast dependency and virtual environment management)
- **LangChain** (loaders, text splitters, prompting, and LLM integrations)
- **ChromaDB** (local vector database)
- **Sentence-Transformers** (using the `all-MiniLM-L6-v2` model)
- **OpenRouter API** (for LLM inference)

## How It Works

```text
PDFs
→ Chunking
→ Embeddings
→ ChromaDB
→ Retrieval
→ LLM Response
```

## Setup

1. **Configure environment**: Create a `.env` file from `.env.example` and provide your OpenRouter API key.
2. **Add documents**: Create a `data/` directory at the project root and place your PDFs inside it.
3. **Run the application**: Use `uv` to start the CLI:
   ```bash
   uv run python src/main.py
   ```

## Example Usage

```text
$ uv run python src/main.py
Initializing RAG CLI application...
Loaded cached document state snapshot successfully.
Index is up to date: No modifications detected in the PDF files.
Connecting to vector database...

=============== PDF RAG CHAT LOOP STARTED ===============
Type your question below. Type 'exit' or 'quit' to end the session.
===========================================================

Ask a question:
> What is the main finding in the paper?

==================== RAG ANSWER ====================
{
    "answer": "The authors find that embedding-based retrieval accuracy increases by 12% when using domain-specific chunking structures.",
    "sources": [
        {
            "source_file": "paper.pdf",
            "page_number": 4,
            "score": 0.8124,
            "preview": "Our experiments demonstrate that applying structural chunking strategies..."
        }
    ],
    "confidence": 0.8124,
    "context": "..."
}
====================================================

Ask a question:
> exit
Exiting chat. Goodbye!
```

## Project Notes

- **Document Type**: Works best with articles, papers, and long-form text documents.
- **Layout Limitations**: Highly formatted sheets, like CVs/resumes or multi-column layouts, may not yield optimal chunking or retrieval results.
- **Educational Intent**: This repository is designed for local prototyping and learning. It implements a basic clean-rebuild indexing strategy on startup rather than real-time synchronization.

## Learning Outcomes

- Learned to build a modular end-to-end RAG pipeline.
- Explored chunking trade-offs with recursive text splitters.
- Handled offline vector generation and local persistence using ChromaDB.
- Integrated OpenRouter APIs into a retrieval-augmented generation workflow.
- Developed basic state tracking to keep files and DB index in sync.
