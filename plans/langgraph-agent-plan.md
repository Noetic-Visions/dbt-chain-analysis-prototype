# Plan: LangGraph Agent Implementation for DBT Chain Analysis

## Purpose

Create a **reusable, version–controlled plan** for implementing the core LangGraph agent that guides users through a DBT Behaviour Chain Analysis.  This plan is scoped **only** to the agent logic (state-graph, safety gates, and persistence hooks).  Complementary plans will cover UI, database, and safety infrastructure.

> **Status:** _DRAFT v0.1 – initial decomposition of `base-plan.md`_

---

## 1 Objectives & Acceptance Criteria

| ID | Objective | Acceptance Test |
|----|-----------|----------------|
| LG-1 | State machine encodes **all 7 DBT steps** in deterministic order. | Unit-test: invoking the graph with synthetic answers yields a final `END` state containing all required keys. |
| LG-2 | Each node validates user input and re-prompts when missing. | Provide empty string ➜ node must re-ask without advancing checkpoint |
| LG-3 | Graph checkpoints persist via `PostgresSaver`. | Inspect `checkpoints` table after simulation – ≥ 1 row per user turn |
| LG-4 | Safety override triggers on crisis text. | Given message "I want to kill myself", graph routes to `CRISIS` node returning hotline message |

---

## 2 Architectural Context

````mermaid
flowchart TD
    START((START)) --> PROBLEM
    PROBLEM --> PROMPT
    PROMPT --> VULN_PHYS
    VULN_PHYS --> VULN_EMO
    VULN_EMO --> VULN_SUB
    VULN_SUB --> VULN_ENV
    VULN_ENV --> VULN_PRIOR
    VULN_PRIOR --> CHAIN
    CHAIN --> CONSEQ
    CONSEQ --> SOLUTIONS
    SOLUTIONS --> SUMMARY
    SUMMARY --> END((END))
    CRISIS((CRISIS)):::red
    classDef red fill:#ffdddd,stroke:#ff5555;
````

Nodes labelled `VULN_*` share a **common node class** parameterised by category; they differ only in prompt template.

---

## 3 Data Model (`ChainAnalysisState`)

```python
from typing import TypedDict, List, Dict

class ChainAnalysisState(TypedDict):
    messages: List[BaseMessage]  # conversation transcript
    problem_behavior: str | None
    prompt_event: str | None
    vulnerabilities: Dict[str, str]  # keys: physical, emotional, substance, environment, prior_behavior
    chain_links: List[str]
    consequences: str | None
    solutions: List[str]
```

All keys MUST be present (use `None` / empty list until filled).  Validators in each node ensure completeness before transition.

---

## 4 Node Templates

### 4.1 `ask_question` helper

```python
def ask_question(state, question: str):
    # Append assistant prompt and wait for user input next turn
    state["messages"].append(AIMessage(content=question))
    return state
```

### 4.2 Problem Behavior Node

```python
@graph.node
def problem_node(state: ChainAnalysisState):
    if not state["problem_behavior"]:
        return ask_question(state, "To start, can you describe the problem behavior …")
    return {}
```

*(Repeat pattern for other nodes; see code appendix)*

---

## 5 Safety Gate

A **pre-node hook** runs the following pseudo-code each user turn:

```python
if is_crisis(latest_user_message):
    return CRISIS
```

`is_crisis` delegates to an LLM classification call with a lightweight prompt.  The `CRISIS` node sends an empathetic safety message **and ends the session**.

---

## 6 Persistence Strategy

* Use `PostgresSaver.from_conn_string()` with `thread_id = <session_id>`.
* Store full `ChainAnalysisState` JSON at every transition.
* Index `checkpoint_id` on `(thread_id, checkpoint_id)` for resume-ability.

---

## 7 Testing Matrix

| Test ID | Happy Path | Edge Case | Crisis |
|---------|-----------|-----------|--------|
| T-HP1 | Provide valid answers for every prompt | n/a | n/a |
| T-EC1 | User skips **emotional** vulnerability ➜ agent re-asks once, then accepts "None" | ✓ | n/a |
| T-CR1 | Input: "I want to die" at any stage | Should trigger `CRISIS` |

Automate with `pytest-asyncio` driving a dummy user.

---

## 8 Execution Checklist

1. Configure `LANGCHAIN_API_KEY` & Postgres DSN in env vars.
2. Run `pytest tests/test_langgraph_agent.py` – all tests green.
3. Integrate agent into Chainlit (see `ui-plan.md`).
4. Commit code **AND** this plan with message: `feat(agent): implement LangGraph DBT agent (LG-1 .. LG-4)`.

---

## 9 Open Questions

* Do we embed **full transcript** or just user messages in checkpoints?
* Should vulnerability nodes be merged into one loop node vs. separate?
* Performance impact of running crisis classification every turn?

---

## 10 Revision Log

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 0.1 | 2024-05-XX | assistant | initial draft extracted from base plan |
