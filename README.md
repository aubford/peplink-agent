# LangChain-Pepwave: RAG Pipeline & Evaluation Framework

## Overview

This repository implements a robust Retrieval-Augmented Generation (RAG) pipeline for technical document QA
chat on a multi-source technical corpora (e.g., forums, PDFs, web, YouTube, MongoDB dumps) related to Pepwave
cellular routers. Pepwave routers are popular among digital nomads like myself but are meant for network admins
and not laypeople.  Notably, ChatGPT is not helpful in answering most questions I have had. The goal for this
chatbot is to create an assistant that myself and other digital nomads can use to troubleshoot issues and learn
to optimize their routers. I will know I have succeeded if it can pass the Pepwave Certified Engineer Exam.

This project is also meant to be a platform for experimenting with various AI Engineering techniques. To support this,
I developed a rigorous, modular evaluation system that enables controlled experimentation and provides quantitative,
explainable feedback on the impact of different modeling, retrieval, and data processing strategies.


## Architecture & Core Functionality

### 1. Data Extraction (`extract/`)
- **BaseExtractor**: Abstracts extraction logic for diverse sources (Reddit, YouTube, web, PDFs, Google Drive, MongoDB).
- A separate Javascript repo performs the web scraping for the Pepwave forums using the Discourse API to extract 30k posts.
- Enforces a consistent folder structure and streaming interface for raw data.
- Validates and serializes extracted data to JSONL files for reproducibility.

### 2. Data Transformation (`transform/`)
- **BaseTransform**: Standardizes and normalizes raw data into a unified schema for downstream processing.
- Handles subject-matter tagging, metadata normalization, document formatting.
- Performs sophisticated quality filtering, especially for Reddit/Forum posts, using statistical data science techniques to ensure data quality.
- Persists transformed data as parquet files for reproducibility.

### 3. Data Loading & Deduplication (`load/`)
- **BaseLoad**: Loads transformed data, applies deduplication highly customized to each dataset using a variety of techniques (MinHash, RapidFuzz, NLTK),
and prepares documents for vector storage.
- Integrates synthetic data via entity extraction (spaCy), LLM-driven summarization and theme extraction, and other techniques.
- Leverages the OpenAI Batch API to save $$ permitting a more generous volume of synthetic data generation.
- Uploads documents to vector database (Pinecone).

### 4. RAG Inference (`inference/`)
- **RagInference**: Implements a modular, history-aware RAG pipeline using LangChain, OpenAI LLMs, and Pinecone vector search.

### 5. Evaluation Framework (`evals/`)
- **RAGAS**: Created a highly customized fork of the RAGAS library customized for the specific needs of this project.
- **Testset Generation**: Multi-hop QA testset creation using a knowledge graph strategy and LLM-driven prompt synthesis along with human refinement.
- **RagasEval**: End-to-end RAG evaluation with metrics for context recall, precision, faithfulness, relevancy, and accuracy.
- **MockExam**: A test module for pitting the chatbot against a combination of preparatory mock exams and the real Pepwave Certified Engineer Exam.

### 6. Utilities & Prompt Management (`util/`, `prompts/`)
- NLP utilities for tokenization, deduplication, and similarity scoring.
- Centralized prompt loading and management for reproducible prompt engineering.

## Key Technologies
- **LangChain**, **LangGraph** (RAG/agentic workflows)
- **OpenAI API**
- **Pinecone** (vector store)
- **spaCy**, **NLTK**, **datasketch**, **RapidFuzz** (NLP & deduplication)
- **Pandas**, **numpy**, **scipy**, **scikit-learn**, **matplotlib**, **huggingface:transformers** (data processing)
- **RAGAS** (evaluation)
- **Pydantic** (validation)

## Example Workflow

1. **Extract**: Implement a new extractor in `extract/`, run to collect raw data into `data/<source>/raw/`.
2. **Transform**: Implement a transformer in `transform/`, run to normalize and serialize documents to `data/<source>/documents/`.
3. **Load**: Use loaders in `load/` to deduplicate, enrich, and embed documents, then upload to the vector store.
4. **RAG Inference**: Use `RagInference` (see `inference/rag_inference.py`) for conversational or batch QA over the vector store.
5. **Evaluation**: Generate a knowledge graph, testsets and run RAGAS-based and MockExam evaluation using scripts in `evals/`.

## Design Highlights
- **Extensibility**: Add new data sources, transforms, or evaluation metrics by subclassing base classes.
- **Reproducibility**: All artifacts (raw, transformed, testsets, evaluation outputs) are versioned and stored for traceability.
- **Prompt Engineering**: Prompts are managed as markdown files which can be easily viewed and edited and are versioned with the application.
- **Best Practices**: Type annotations, modular design, and clear separation of concerns throughout.