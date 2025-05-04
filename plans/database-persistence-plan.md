# Plan: PostgreSQL Persistence & Schema for DBT Chain Analysis Assistant

## Purpose

Define the relational schema, migrations, and access patterns for persisting conversation data, checkpoints, and analysis summaries in PostgreSQL.  This plan intentionally excludes application code (handled in agent/UI plans) and focuses on DDL, indexing, and migration tooling.

> **Status:** _DRAFT v0.1_

---

## 1 Objectives & Acceptance Criteria

| ID | Objective | Acceptance Test |
|----|-----------|----------------|
| DB-1 | Core tables (`users`, `sessions`, `messages`, `chain_analysis_results`) created via Alembic migration. | Run `alembic upgrade head`; tables exist with correct columns & constraints |
| DB-2 | `checkpoints` table managed by LangGraph `PostgresSaver` coexists without FK conflicts. | Insert checkpoint during agent simulation – no FK errors |
| DB-3 | `messages` table stores embeddings column (`vector(384)`) when pgvector extension enabled. | After insert, vector column not null (if extension present) |

---

## 2 Schema Diagram

```mermaid
erDiagram
    users ||--o{ sessions : has
    sessions ||--|{ messages : contains
    sessions ||--|| chain_analysis_results : "produces"
    sessions ||--o{ checkpoints : "lg checkpoints" 

    users {
        uuid id PK
        text email
        text name
        timestamptz created_at
    }
    sessions {
        uuid id PK
        uuid user_id FK
        text type
        timestamptz started_at
        timestamptz ended_at
    }
    messages {
        bigint id PK
        uuid session_id FK
        text role
        text content
        vector(384) embedding
        timestamptz ts DEFAULT now()
    }
    chain_analysis_results {
        uuid session_id PK FK
        jsonb vulnerabilities
        jsonb chain_links
        text problem_behavior
        text prompting_event
        text consequences
        jsonb solutions
        text summary
        timestamptz created_at
    }
```

---

## 3 Migration Strategy

* Use **Alembic**.  Directory structure:
  * `migrations/versions/*.py`
* `env.py` loads `DATABASE_URL` from env.
* CI job runs `alembic upgrade head` in `pytest` workflow to ensure migrations apply.

### 3.1 Initial Migration Stub

```python
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgvector")

    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('email', sa.Text(), unique=True),
        sa.Column('name', sa.Text()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now())
    )
    # ... repeat for other tables ...
```

---

## 4 Indexing & Performance

| Table | Column(s) | Index Type | Rationale |
|-------|-----------|------------|-----------|
| `messages` | `session_id, ts` | BTREE | Fast retrieval of conversation by session chronologically |
| `messages` | `embedding` | IVFFlat (pgvector) | Semantic search on past content |
| `checkpoints` | `thread_id, checkpoint_id` | BTREE | Resume latest checkpoint quickly |

---

## 5 Access Layer

Implement **asyncpg** wrapper in `app/db.py`:

```python
class DB:
    def __init__(self, dsn):
        self.pool: asyncpg.Pool = await asyncpg.create_pool(dsn)

    async def execute(self, sql, *args):
        async with self.pool.acquire() as conn:
            await conn.execute(sql, *args)
```

(Full code in implementation task, not in plan.)

---

## 6 Testing Matrix

| Test | Steps | Expected |
|------|-------|----------|
| T-DB-1 | Spin up ephemeral Postgres via `pytest-postgresql`; run migrations | All tables exist |
| T-DB-2 | Insert dummy conversation; query back | Values match |
| T-DB-3 | Upsert embedding & run similarity search | Returns expected row |

---

## 7 Open Questions

* Should we store **per-link timestamps** in `chain_links`?
* Retention / GDPR purge strategy?

---

## 8 Revision Log

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 0.1 | 2024-05-XX | assistant | initial draft |

