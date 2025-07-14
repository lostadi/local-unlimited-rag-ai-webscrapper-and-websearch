# AI Web Search RAG Assistant

## Overview

AI Web Search RAG Assistant is a self‑hosted Retrieval‑Augmented Generation (RAG) toolkit that pairs a local SearXNG metasearch engine with open‑weights language and embedding models served by Ollama. It can search the live web, scrape relevant pages, build a FAISS knowledge base, and answer questions through either a Gradio Web UI or an interactive CLI.

Key capabilities:

* **Web discovery** via local SearXNG API.
* **Polite scraping** with adaptive delays and user‑agent rotation.
* **Chunking and vector storage** using LangChain + FAISS.
* **LLM answer synthesis** with open models (defaults: `huihui_ai/qwen3‑abliterated:4b` and `nomic‑embed‑text`).
* **Session persistence**: save / load knowledge bases to `~/my_rag_shared_contexts`.
* **Two front ends**: Gradio UI (`app_gradio_rag.py`) and terminal CLI (`app_cli_rag.py`).

## Quick Start

```bash
# 1. Clone or copy the repo
git clone <repo_url>
cd ai_web_search_engine   # directory containing this README

# 2. Run the one‑shot installer
chmod +x inde_search_llm.sh
./inde_search_llm.sh
```

The script will:

1. Verify system dependencies (`python3`, `pip`, `ollama`, `curl`, `git`).
2. Pull the default Ollama models.
3. Install Python packages (`gradio`, `langchain`, etc.).
4. Copy and patch the project’s Python modules into `~/ai_rag_apps/`.

### Launch the Web UI

```bash
cd ~/ai_rag_apps
python3 app_gradio_rag.py
# then open http://localhost:12123
```

A desktop shortcut is also created automatically on Linux desktops.

### Use the CLI

```bash
cd ~/ai_rag_apps
python3 app_cli_rag.py
```

Type `search`, `query`, `save`, `load`, or `exit` when prompted.

## Architecture

```text
┌────────────┐       query        ┌───────────┐     scraped HTML      ┌────────────┐
│   Gradio   │ ───────────────▶ │  SearXNG  │ ───────────────────▶ │   Scraper   │
│    UI      │ ◀─────────────── │  (local)  │ ◀─────────────────── │ (requests) │
└────────────┘   answer + cite   └───────────┘   content snippets    └────────────┘
      ▲                                                           │
      │                                               FAISS index │
      │                                                           ▼
┌────────────┐   retrieved docs   ┌──────────────┐   embeddings   ┌─────────────┐
│    CLI     │ ───────────────▶ │   Vector DB   │ ◀────────────── │ Embeddings  │
└────────────┘ ◀─────────────── │    (FAISS)   │                 └─────────────┘
                                 └──────────────┘
                                          │
                                          ▼
                                 ┌────────────────┐
                                 │   Ollama LLM   │
                                 └────────────────┘
```

Core logic lives in **`rag_core.py`** and is imported by both front‑ends, ensuring a single source of truth for configuration and RAG utilities.

## Configuration

Most knobs can be tweaked in `rag_core.py`:

| Setting                      | Default                          | Description                           |
| ---------------------------- | -------------------------------- | ------------------------------------- |
| `SEARXNG_URL`                | `http://localhost:8080`          | Location of your SearXNG instance     |
| `OLLAMA_MODEL_PY`            | `huihui_ai/qwen3-abliterated:4b` | LLM to generate answers               |
| `OLLAMA_EMBEDDING_MODEL_PY`  | `nomic-embed-text`               | Embedding model for vector store      |
| `CHUNK_SIZE / CHUNK_OVERLAP` | 1000 / 200                       | Text splitter granularity             |
| `K_RETRIEVER`                | 200                              | How many chunks to retrieve per query |
| `K_CONTEXT_FOR_LLM`          | 50                               | How many chunks to send to the LLM    |

The installer also allows you to override:

* **Gradio port** (`GRADIO_PORT`, default 12123).
* **Target app directory** (`APP_DIR`, default `~/ai_rag_apps`).
* **Model names** via `OLLAMA_MODEL` and `OLLAMA_EMBEDDING_MODEL` shell variables.

## Saving & Loading Contexts

Both UI and CLI can persist knowledge bases:

* **Gradio**: use “Save Context” / “Load Context” buttons.
* **CLI**: `save` and `load` commands.

Artifacts are stored as a FAISS index plus JSON metadata under `~/my_rag_shared_contexts/`.

## Troubleshooting

| Symptom                                   | Likely Cause                     | Fix                                                              |
| ----------------------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| `Could not connect to local SearXNG…`     | SearXNG not running or wrong URL | Start SearXNG and confirm `SEARXNG_URL`                          |
| `Connection refused` when pulling models  | Ollama daemon not running        | `ollama serve`                                                   |
| Gradio launches but returns empty answers | No context scraped               | Verify your query returns results or raise `SEARXNG_MAX_RESULTS` |

Verbose logging is printed to the terminal for each component (discover, scrape, vector store, generation).

## Folder Layout (after install)

```
ai_rag_apps/
├── app_gradio_rag.py   # Web UI entrypoint
├── app_cli_rag.py      # Command‑line entrypoint
├── rag_core.py         # Shared RAG engine
└── ...
```

## Roadmap

* Plug‑in authentication for private search hosts
* Async scraping for speed
* UI theming and citations preview

## License

My Driver's License 

## Credits

Built with LangChain, FAISS, Gradio, Ollama, and SearXNG.
