# llm-qa-with-langgraph

A lightweight Streamlit application that uses LangGraph to maintain conversational Q&A history while calling a model through OpenRouter.

## Overview

This project demonstrates a simple sequential workflow where each user question is processed by a LangGraph node and appended to a stateful history list. The app keeps prior questions and answers in memory during the session and uses a configurable model from OpenRouter.

## Features

- Stateful Q&A workflow using LangGraph
- Persistent chat history within a Streamlit session
- Configurable model selection through environment variables
- Simple OpenRouter integration for LLM access

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   ├── main.py
│   └── llm_qa_with_langgraph/
│       └── __init__.py
└── .env
```

## Requirements

- Python 3.12+
- An OpenRouter API key

## Setup

1. Install `uv` if you do not already have it:

```bash
pip install uv
```

2. Create the project environment and install dependencies with `uv`:

```bash
uv sync
```

This creates and manages the Python environment for the project using `uv`, without needing to create a virtual environment manually.

3. Create a `.env` file in the project root with your OpenRouter credentials:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

4. Run the app with `uv`:

```bash
uv run streamlit run src/main.py
```

If you prefer to activate the environment manually, you can still do so after `uv sync`:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
streamlit run src/main.py
```

## How it works

- The app loads environment variables from `.env`.
- A `StateGraph` workflow is created with a single node that sends the newest question to the selected OpenRouter model.
- The model response is stored alongside the question in the workflow state.
- Streamlit renders the conversation history and appends new answers as users continue asking questions.

## Notes

- This app is intentionally minimal and designed as a learning/demo project for LangGraph-based LLM workflows.
- You can swap the model name in `OPENROUTER_MODEL` to use different providers or model families supported by OpenRouter.
