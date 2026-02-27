# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pepwave Tech Support RAG Chatbot — a Python 3.12 FastAPI + LangGraph application providing conversational Q&A about Pepwave cellular routers using Pinecone vector search, OpenAI LLMs, and Cohere reranking.

### Running the web app (dev mode)

**With PostgreSQL (preferred — persistent conversations):**

```bash
# Start PostgreSQL (Docker must be running: sudo dockerd if not)
docker compose up -d postgres

# Start the web app
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable"
export USE_IN_MEMORY_CHECKPOINTER=false
PYTHONPATH=/workspace langchain_pepwave_env/bin/python -m uvicorn web_app.app:app --reload --host 0.0.0.0 --port 8000
```

**Without PostgreSQL (in-memory, state lost on restart):**

```bash
export USE_IN_MEMORY_CHECKPOINTER=true
PYTHONPATH=/workspace langchain_pepwave_env/bin/python -m uvicorn web_app.app:app --reload --host 0.0.0.0 --port 8000
```

**Gotcha:** The environment may pre-set `USE_IN_MEMORY_CHECKPOINTER=false`. Since `python-dotenv` does not override existing env vars, you must explicitly `export` the variable before starting the server. If PostgreSQL is not running and `USE_IN_MEMORY_CHECKPOINTER` is not `true`, the app will crash on startup with a connection error.

**Gotcha:** Docker in this cloud VM requires `sudo dockerd` to start the daemon, and `sudo chmod 666 /var/run/docker.sock` for non-root access.

### Required API keys (environment secrets)

- `OPENAI_API_KEY` — LLM inference + embeddings
- `PINECONE_API_KEY` — vector store retrieval
- `COHERE_API_KEY` — document reranking
- `TAVILY_API_KEY` (optional) — web search tool
- `LANGSMITH_API_KEY` (optional) — tracing/observability

### Linting

```bash
langchain_pepwave_env/bin/flake8 --max-line-length=120 --exclude=langchain_pepwave_env,data,.git web_app/ inference/ config/ tests/
langchain_pepwave_env/bin/black --check web_app/ tests/
```

Existing code has pre-existing style issues (long lines, quote style). These are not regressions.

### Testing

```bash
PYTHONPATH=/workspace langchain_pepwave_env/bin/python -m pytest tests/ -v
```

- `test_deduplication.py` (2 tests) — pass
- `test_reddit_transform.py` (6 tests) — pre-existing failures due to constructor mismatch (`RedditTransform()` called without required args)

### Key paths

| Path | Description |
|------|-------------|
| `web_app/` | FastAPI app (port 8000) |
| `inference/` | LangGraph RAG pipeline |
| `config/` | Config singleton, logger |
| `tests/` | pytest tests |
| `prompts/` | Markdown prompt templates |
| `Makefile` | Convenience targets (see `make help`) |

### Virtual environment

The venv is at `langchain_pepwave_env/`. Always use `PYTHONPATH=/workspace` when running Python commands so project-root imports resolve correctly.
