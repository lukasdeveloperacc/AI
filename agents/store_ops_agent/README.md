# StoreOpsAgent

AI-powered store operations assistant using LangGraph and FastAPI.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

## Installation

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup project

```bash
# Clone the repository
git clone <repository-url>
cd store_ops_agent

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your configuration:

```
OPENAI_API_KEY=your-openai-api-key-here
FAISS_INDEX_PATH=./data/index
DOCSTORE_PATH=./data/documents
LOG_LEVEL=INFO
```

## Running the Application

```bash
uv run uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`. Access the Swagger UI at `http://localhost:8000/docs`.

## Project Structure

```
store_ops_agent/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application entry point
│   ├── api/              # API routes
│   ├── core/             # Core configuration and utilities
│   ├── agents/           # LangGraph agents
│   ├── indexing/         # Document indexing and vector store
│   └── models/           # Pydantic models
├── data/
│   ├── documents/        # Source documents
│   └── index/            # FAISS index files
├── tests/                # Test files
├── .env.example          # Environment variable template
├── pyproject.toml        # Project dependencies (uv)
├── uv.lock               # Locked dependencies
└── README.md
```

## Development

### Running tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## License

MIT
