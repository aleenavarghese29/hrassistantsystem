# HR Assistant RAG System

## Overview

This project builds an **AI-powered HR Assistant** that can answer employee questions about company policies.
It uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant information from HR policy documents and generate responses.

The system processes policy documents, converts them into embeddings, stores them in a vector database, and retrieves relevant information when employees ask questions.

---

# Tools & Technologies Used

* **Python**
* **FastAPI** – API framework for the HR assistant
* **FAISS** – Vector database for embedding storage and similarity search
* **Ollama (Llama 3)** – Language model for generating responses
* **Sentence Transformers (all-MiniLM-L6-v2)** – Embedding model
* **Uvicorn** – ASGI server for running the FastAPI application

---

# Project Structure

```
hr_assistant/
│
├── applications/
│   ├── app.py                # FastAPI API
│   └── rag_system.py        # RAG pipeline implementation
│
├── ingestion/
│   ├── clean_docs.py        # Document cleaning
│   ├── clean_texts.py       # Text preprocessing
│   ├── chunk_text.py        # Document chunking
│   └── generate_embeddings.py # Embedding generation & FAISS index
│
└── requirements.txt
```

---

# Stage 1 — Current Progress

Implemented the **core RAG pipeline**.

### Data Processing

Policy documents are processed through several steps:

1. **Cleaning documents**
2. **Preprocessing text**
3. **Chunking large documents**
4. **Generating embeddings**
5. **Storing embeddings in FAISS**

### RAG System

The system retrieves relevant document chunks using **semantic search** and generates responses using **Llama 3 via Ollama**.

### API

A **FastAPI backend** exposes the HR assistant so users can send questions and receive responses.

Run the API:

```
uvicorn applications.app:app --reload
```

---

# Stage 2 — Planned Development

Next stage will introduce **employee data and HR task automation**.

Planned features:

* Employee database
* Storing employee information
* Leave balance tracking
* Leave application system
* HR workflow automation

The assistant will evolve from **question answering → task execution**.

Example:

```
Employee: Apply leave tomorrow
Assistant: Checking leave balance...
Assistant: Leave applied successfully.
```

---

# Stage 3 — Deployment

Final stage focuses on making the system production-ready.

Planned work:

* Deploy API
* Containerization with Docker
* Cloud deployment
* Authentication and role-based access
* Building a complete **agentic HR assistant**

---

# Author

Aleena Varghese
