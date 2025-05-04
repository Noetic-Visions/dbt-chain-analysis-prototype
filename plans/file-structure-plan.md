# Plan: Proposed File Structure for DBT Chain Analysis Prototype

## Purpose

This document outlines the proposed directory and file structure for the `dbt-chain-analysis-prototype` project, based on the requirements derived from the other planning documents (`base-plan.md`, `chainlit-ui-plan.md`, `database-persistence-plan.md`, `langgraph-agent-plan.md`).

## Proposed Structure

```
dbt-chain-analysis-prototype/
├── app/                      # Core application logic
│   ├── __init__.py
│   ├── agent.py              # LangGraph agent definition, nodes, state
│   ├── db.py                 # Database connection pool, query helpers (asyncpg wrapper)
│   ├── main.py               # Chainlit application entry point (@on_chat_start, @on_message)
│   └── models.py             # Pydantic or dataclass models (e.g., for state, DB results)
│
├── migrations/               # Alembic database migrations
│   ├── versions/             # Individual migration scripts
│   ├── env.py                # Alembic environment configuration
│   └── script.py.mako        # Alembic script template
│
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   ├── test_agent.py         # Tests for LangGraph agent logic
│   ├── test_chainlit_ui.py   # Tests for Chainlit handlers (using test client)
│   ├── test_db.py            # Tests for database interactions and schema
│   └── fixtures/             # Test fixtures (e.g., sample data)
│
├── plans/                    # Project planning documents (existing)
│   ├── base-plan.md
│   ├── chainlit-ui-plan.md
│   ├── database-persistence-plan.md
│   ├── langgraph-agent-plan.md
│   └── file-structure-plan.md # This proposed plan
│
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── alembic.ini               # Alembic configuration file
├── pyproject.toml            # Project metadata and dependencies (using Poetry or similar)
└── README.md                 # Project overview, setup, and usage instructions
```

## Rationale

1.  **`app/`**: Centralizes the main application code.
    *   `agent.py`: Houses the LangGraph state machine, node definitions, and `ChainAnalysisState` as per `langgraph-agent-plan.md`.
    *   `db.py`: Contains the database interaction logic (asyncpg wrapper, query functions) outlined in `database-persistence-plan.md` and `chainlit-ui-plan.md`.
    *   `main.py`: The Chainlit entry point, containing `@cl.on_chat_start` and `@cl.on_message` handlers as described in `chainlit-ui-plan.md`.
    *   `models.py`: Useful for defining shared data structures like the `ChainAnalysisState` TypedDict or Pydantic models for database interactions, promoting consistency.
2.  **`migrations/`**: Standard Alembic directory structure for managing database schema changes as specified in `database-persistence-plan.md`.
3.  **`tests/`**: Dedicated directory for tests, separated by component (`agent`, `ui`, `db`) for clarity, following best practices mentioned across the plans.
4.  **`plans/`**: Keeps all planning documents together, including this new structure plan.
5.  **Root Files**: Standard project configuration files (`.env.example`, `.gitignore`, `pyproject.toml`, `README.md`) and Alembic config (`alembic.ini`).
