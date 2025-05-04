# Implementation Plan: AI Chain Analysis Assistant

## Overview and Goals

This plan outlines a comprehensive design for an AI system that guides Dialectical Behavior Therapy (DBT) clients through a Behavior Chain Analysis. The goal is to help users identify day-specific vulnerability factors and other chain analysis steps in a conversational manner. The system will be built with Python 3.12, leveraging LangGraph for agent workflow and state management, Chainlit for the user interface, and a PostgreSQL database for persistent memory of user interactions. By integrating these components, the AI assistant will autonomously lead users through the structured steps of a chain analysis whenever a user requests help, focusing on:

*   Defining the problem behavior
*   Identifying the prompting event
*   Exploring vulnerability factors (specific to that day)
*   Mapping out the chain of internal/external links (events, thoughts, feelings, actions)
*   Discussing consequences, "absent aids" (missed coping strategies), and brainstorming solutions

This design follows best practices from Anthropic's "Building Effective Agents" framework and the LangGraph documentation to ensure the agent is reliable and safe. The system will maintain a supportive, empathetic tone while adhering to safety guidelines for sensitive mental health topics. In the following sections, we detail the system architecture, the agent's conversation flow, memory management, prompting strategy, and integration details.

## System Architecture and Components

### Tech Stack

The system combines a conversational AI model with supporting frameworks: Python 3.12 as the development language, LangGraph for orchestrating the agent's behavior, Chainlit for the front-end chat UI, and PostgreSQL for persisting data. The diagram below illustrates the high-level architecture and data flow between components:

**Figure:** Finite-state conversation flow for the Chain Analysis agent. Each step corresponds to a stage in the DBT chain analysis (Problem Behavior, Prompting Event, Vulnerabilities, etc.), modeled as states in a LangGraph workflow. The agent transitions through these states sequentially, prompting the user for input at each stage. The vulnerability factors step (Step 3) expands into sub-questions about physical, emotional, substance-related, environmental, and behavioral factors (dashed connections), ensuring day-specific vulnerabilities are thoroughly examined.

*   **User & Chainlit UI:** Users interact with the AI via a Chainlit-powered web interface. Chainlit provides a chat interface with streaming support, user sessions, and integration hooks. When a user initiates a chain analysis (e.g. by typing a request like "Help me analyze yesterday's incident"), Chainlit will trigger the backend logic.
*   **LangGraph Agent:** The core AI logic is implemented as a LangGraph `StateGraph` (a form of finite state machine or directed acyclic graph for conversation) controlling an LLM. Each node in the graph represents a step in the chain analysis workflow (problem definition, prompt event, each vulnerability factor prompt, etc.). LangGraph allows us to define transitions between these steps and to maintain an internal state (memory) across the multi-turn dialogue. This design ensures the conversation follows the structured DBT workflow reliably, rather than leaving all control to the LLM. (Anthropic's guidance suggests using simple, composable patterns and predefined workflows for well-defined tasks – our approach aligns with this by treating chain analysis as a structured workflow, not an unconstrained, unpredictable agent.)
*   **LLM (Large Language Model):** We will use a capable LLM (such as GPT-4, Claude, or another model accessible via LangChain) to generate the conversational content. The LLM is prompted at each state to produce the next question or to summarize user input, following the DBT chain analysis script. LangGraph's integration with LangChain allows tool use if needed, but in this case the primary "tool" is the conversation itself. We might incorporate moderation or analysis tools (for safety checks or sentiment analysis) as part of the workflow if needed (see Safety section below).
*   **PostgreSQL Database:** A Postgres database serves as a persistent memory store for chat history and user session data. It will record user messages, the assistant's responses, and structured outcomes of each chain analysis. This enables the system to remember past sessions and personalize future interactions. LangGraph provides a Postgres persistence layer for conversation state ("checkpoints"), which we will use to save and reload sessions . Storing data in Postgres ensures that even if the session is restarted or the user returns later, the AI can retrieve prior context (with user consent), maintaining continuity. We also plan to use the Postgres DB (with extensions like pgvector) to store vector embeddings of conversation summaries for semantic search of historical context if needed. This means the assistant can recall relevant details from previous chain analyses to avoid repetition and to identify patterns over time.

Below is a summary of how these components interact at runtime:

1.  **Session Start:** When a user opens the Chainlit chat for this assistant, a new session (or thread) is created. In `@cl.on_chat_start`, the backend will initialize the LangGraph state machine for the chain analysis agent and establish a database connection. For example, we might assign a unique `session_id` or use the user's ID to identify the conversation thread. Using LangGraph's `PostgresSaver`, we attach the database to the agent's state, so that any state changes are automatically saved as checkpoints . This ensures the conversation can be paused/resumed seamlessly. If historical context exists (e.g. the user has done past analyses), we can fetch a summary of relevant history from Postgres here and prime the agent's context with that information (more on this in the Memory section).
2.  **User Input & Agent Response Cycle:** For each user message, the Chainlit `@cl.on_message` event is triggered. The backend will retrieve the LangGraph agent instance from the `cl.user_session` (stored at chat start) and invoke the agent with the new user message. Internally, the LangGraph `StateGraph` will append the user input to the conversation state and progress to the appropriate next node. The LLM then generates a response according to that node's logic. Chainlit's callback handler streams the LLM's response back to the UI in real-time, so the user sees the assistant "typing" the answer. The response is also logged to Postgres. This cycle repeats, traversing the state graph step by step.
3.  **Session End / Persistence:** When the chain analysis conversation concludes (either the user finishes all steps or exits), the final state (including all collected answers and any summaries) is saved to the database. The user's session history remains available for future retrieval. If the user starts another chain analysis later, the agent can query this stored data to inform its responses (for instance, noting if a certain vulnerability factor frequently recurs for this user).

By combining Chainlit (for a rich UI and session management) with LangGraph's persistent state and Postgres, we achieve both an intuitive user experience and robust data handling on the backend. The agent's design now needs to reflect the DBT chain analysis structure, which we detail next.

## LangGraph Agent Design (Workflow & Memory)

The conversation with the AI will closely mirror the structured steps of a DBT Behavior Chain Analysis. We design the agent as a guided workflow (an FSM/DAG) with a predetermined sequence of states, ensuring all key steps are covered in order. Unlike a free-form chat, this structured approach guarantees consistency and completeness in the analysis, aligning with Anthropic's recommendation that well-defined tasks benefit from workflow-style agents for predictability .

### State Machine Pattern

Using LangGraph, we implement a state graph where each node corresponds to a step or sub-step in the chain analysis. Transitions between nodes are largely linear (following the DBT step order), but we allow for some conditional branching or looping as needed. For example, the agent might loop within the "chain of events" step to ask repeatedly "What happened next?" until the user indicates the sequence is complete. We might also iterate through various vulnerability factors sub-nodes to ensure each category is considered. The figure above (Figure: Chain Analysis Conversation Flow) illustrates this state sequence.

Key states/nodes in the LangGraph design:

1.  **Problem Behavior Definition:** The agent asks the user to describe the problem behavior in detail (Step 1 of chain analysis). This is the initial node after the start. The user's response (which might be a harmful behavior, emotional outburst, etc.) is stored in the state (e.g., `state["problem_behavior"]`). The agent uses follow-up prompts if needed to ensure specificity: "What exactly did you do, what did you feel, and how intense was it?" This establishes the target of analysis.
2.  **Prompting Event Identification:** Next, the agent transitions to the prompting event node. It asks "What was the precipitating event that triggered this chain?" (Step 2). The agent will clarify that this is the environmental or situational event that occurred right before the chain reaction started (even if it seemingly shouldn't cause the behavior). The user's answer (e.g., "I had an argument at work" or "I got a bad news call") is saved as `state["prompt_event"]`. The agent might use sub-questions from DBT worksheets such as "Why did it happen today and not another day?" to pinpoint the timing.
3.  **Vulnerability Factors (Day-Specific):** This is a crucial part (Step 3) where the agent explores what made the user more vulnerable on that day *before* the prompting event. We implement this as either a single state with a composite prompt or a series of sub-states for each vulnerability factor type, to ensure thorough coverage. The areas to examine, as per DBT guidelines, include: physical factors, emotional state, substances, environmental stressors, and prior actions. To handle this, the agent will likely break this down into multiple questions in sequence:
    *   **Physical Vulnerabilities:** "Were there any physical or biological factors that day that left you vulnerable? (For example: illness, pain, fatigue, poor sleep, hunger)" The agent gently queries if the user had any health issues, lack of sleep, skipped meals, etc. The answer is stored (e.g., `state["vulnerability"]["physical"]`).
    *   **Emotional State:** "How were you feeling emotionally before the event? Any intense emotions or mood that made you more susceptible?" High baseline sadness, anxiety, anger, etc., could be noted (`state["vulnerability"]["emotional"]`).
    *   **Substance Use:** "Had you used any substances prior to the event? (This could include alcohol, drugs, or even medications)." Substance use can lower coping ability; any details go to `state["vulnerability"]["substance"]`.
    *   **Environmental Stressors:** "Were there other stressful events or situational stress that day (earlier or ongoing)?" Perhaps work stress, family issues, or other external pressures (`state["vulnerability"]["environment"]`).
    *   **Previous Behaviors:** "Did anything you did earlier that day (or the previous day) contribute to stress or make things worse? For example, an earlier conflict or mistake that left you upset" This captures previous behaviors of your own that you found stressful, a commonly overlooked vulnerability (stored in `state["vulnerability"]["prior_behavior"]`).
    The agent will iterate through these sub-questions one by one, adopting a compassionate and non-judgmental tone for each. If the user's response to a particular category is "None" or they already mentioned it, the agent can gracefully move on. The LangGraph could treat each of these as a sub-node connected in sequence (as shown with dashed arrows under Step 3 in the diagram). After collecting all vulnerability info, the state now contains a detailed picture of why the user was especially vulnerable that day. This addresses the "day-specific" aspect: Why did this happen today instead of another day? The assistant ensures those factors are clearly identified.
4.  **Chain of Events (Links):** At this state, the agent helps the user map out the chain of internal and external events that led from the prompting event to the problem behavior (Step 4). This often involves dissecting the sequence in detail: the thoughts, feelings, action urges, and actions that occurred link by link. We design this node to possibly loop and prompt repeatedly:
    *   The agent might start: "Let's walk through what happened step by step. After the prompting event (<user's event>), what was the first thing you did, thought, or felt?"
    *   The user responds with the first link (e.g., "I started thinking that I was worthless"). The agent stores this in something like `state["chain_links"] = [first_link]`.
    *   Then the agent asks: "What happened next? What did that lead to?" This continues iteratively: each answer appended to the `chain_links` list. The agent ensures to include internal experiences (thoughts, feelings, bodily sensations) as well as actions or external events. If the user struggles, the agent can gently probe different modalities: "What were you feeling at that moment? And how did you respond?"
    *   This looping continues until the user reaches the point of the problem behavior itself (one of the links will be the problem behavior we identified in Step 1). The agent might detect this because the user's description matches the identified behavior, or the user says "and that's when I [problem behavior]." At that point, the chain is complete.
    *   The final outcome is a chronological chain in the state (a list of links). Optionally, the agent can echo back a summary: "So, in summary, the sequence was: [trigger] → [thought] → [feeling] → [urge] → [action] … → [problem behavior]." This ensures the user feels heard and the chain is correctly captured.
5.  **Consequences:** Now the agent asks about the aftermath of the problem behavior (Step 5). It transitions to a node where it inquires: "What happened after the behavior? How did you feel right after, and later on? How did others react, and what effect did it have on your environment or situation?" This helps the user evaluate the impact of the behavior. The assistant will capture both short-term relief or other immediate consequences and long-term negative outcomes. For example, a user might respond: "I felt a bit of relief for a few minutes, but then I felt shame and my partner was very upset with me." These details go into `state["consequences"]` (which could be structured as internal consequences vs external consequences). This step is important for building motivation to change, as it highlights why the behavior is problematic.
6.  **Absent Aids & Solution Brainstorming:** In the final content step, the agent focuses on problem-solving (Step 6, and partially Step 7 from DBT, which covers prevention). This node helps the user reflect on what could have been done differently at various points in the chain, and how to prevent a similar outcome in the future. Specifically:
    *   The agent may guide the user to review the chain and identify points where a different action or skill could have changed the outcome: "Looking back at each link, can you spot places where you could have used a coping skill or done something differently to avoid reaching the harmful behavior?" This corresponds to absent aids – coping strategies or helpful actions that were missing in the original chain. The user might say, for example, "After the argument (prompting event), I could have called a friend instead of isolating" or "When I started thinking I was worthless, I could have used a thought-challenging skill." The agent notes these ideas (possibly annotating the chain links with alternative strategies).
    *   Next, the agent asks to brainstorm general solutions or preventive measures: "What can we do going forward to reduce the chances of this happening again? Are there ways to reduce those vulnerability factors, or to handle the trigger better next time?" This encourages the user to come up with concrete steps (like practicing a "Plan to reduce vulnerability" for next time – e.g., get proper sleep, avoid certain triggers, use mindfulness when emotions run high, etc.). The agent might also contribute suggestions if the user is stuck, drawing on DBT skills knowledge (while being careful not to lecture).
    *   All the brainstormed solutions and prevention strategies are collected in `state["solutions"]` (could be a list of ideas or a mini action plan). The tone remains collaborative and empowering, reinforcing that it's possible to change the chain in future.
7.  **Conclusion and Summary:** Finally, the conversation reaches an end state. The agent thanks the user for their honesty and effort, and offers a concise summary of the chain analysis. For example, it might summarize: "Today, we identified that [Problem Behavior] happened after [Prompting Event]. You were especially vulnerable because [Vulnerabilities]. The chain of events was [Link1 → Link2 → … → Behavior], leading to [Consequences]. Next time, you plan to [Solutions], and reduce vulnerability by [Prevention]. Great work on this analysis." This recap helps solidify the insights. The summary can be stored in the database as a standalone record (for easy reference later). The agent also reminds the user of any follow-up (like practicing certain skills or discussing with their therapist if applicable).

### LangGraph Implementation

In code, we will use LangGraph's `StateGraph` API to define this workflow. Each numbered step above corresponds to a function or class that implements the logic for that node. For example, we might define a `ChainAnalysisState` TypedDict to hold all the pieces of information (problem, prompt_event, vulnerabilities, etc.). Then:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

# Define the state data structure for the chain analysis
class ChainAnalysisState(TypedDict):
    problem_behavior: str
    prompt_event: str
    vulnerabilities: Dict[str, str]   # e.g., {"physical": ..., "emotional": ..., ...}
    chain_links: List[str]
    consequences: str
    solutions: List[str]
    # Potentially add messages list for conversational state
    # messages: List[BaseMessage] 

graph = StateGraph(ChainAnalysisState)  # initialize the state graph
```

We add nodes to the graph for each step. For instance, a `ProblemBehaviorNode` function might prompt the user for the problem behavior if not yet provided, or process the user's description:

```python
def prompt_problem_behavior(state: ChainAnalysisState):
    # LLM prompt to ask for problem behavior if not already in state
    if not state.get("problem_behavior"):
        # The agent asks the question using the LLM:
        question = ("To start, can you describe the problem behavior you want to analyze? "
                    "What happened, and what did you do or feel? Be as specific as you can.")
        # Assume llm is defined and invokable
        # response = llm.invoke(question) 
        # Return the assistant message to be shown to the user
        # This needs integration with how messages are handled in the state
        # Example: return {"messages": [AIMessage(content=question)]}
        # Or maybe a dedicated field for assistant response
        return {"assistant_msg": question} # Simplified for illustration
    return {} # No change if behavior already known
```

However, since our agent will run as a conversation loop, a simpler approach is to let the user supply the answers and the agent to guide with questions. In LangGraph, one pattern is to treat the conversation messages as part of the state and use placeholders (like in a chat bot). For example, we could maintain a `state["messages"]` (the dialogue so far) and have the agent produce the next system/human prompt accordingly. But an easier mental model is to call the LLM with a crafted prompt at each step.

### Finite-State Workflow vs. Dynamic Agent

It's worth noting that our design leans towards a workflow (fixed sequence) rather than a fully dynamic agent. This is intentional: chain analysis has a clear sequence that benefits from structure. Anthropic notes that "prompt chaining decomposes a task into fixed subtasks" and is ideal when the sequence is well-defined. We implement these subtasks as nodes. We do allow some flexibility (e.g., looping until a condition, or skipping certain prompts if the info is already given), but the overall flow is controlled. This ensures reliability and consistency of the output. The LLM still has a lot of freedom in how to phrase questions and empathize with the user, but *what* to ask next is guided by the state machine. In practice, this yields a more effective agent for our use case, as it won't forget steps or get stuck – it has a clear agenda to follow.

### Memory & State Management

As the conversation progresses through states, LangGraph will carry over data in the `ChainAnalysisState`. For example, after the user answers the prompting event, the `prompt_event` field is filled and remains available for the next nodes. We will use LangGraph's persistence to save this state in Postgres at checkpoints. In particular, the `PostgresSaver` utility provides automatic saving of the state after each node (or each user turn) to a `checkpoints` table . Each session (thread) gets a unique ID, and the checkpoint table stores the entire state JSON for that session ID and each step. This means if the session disconnects or the user returns later, we can reload the latest state and continue where we left off . The Postgres persistence effectively gives us short-term memory across turns and long-term memory across sessions. The system can "remember" the user's name, prior answers, or even previous chain analyses because it's stored and indexed by user/session.

For example, if the user's session ID is `user123_session1`, after they describe their problem behavior, the state might look like: `{"problem_behavior": "I self-harmed by cutting", "prompt_event": null, ...}` and is saved to DB. On the next turn, LangGraph loads this state, and now when the user provides the prompting event, it updates to `{"problem_behavior": "...", "prompt_event": "Had an argument with my mother", ...}` and saves again, and so forth. By the end, the final state in the DB will contain the full chain analysis data. This structured memory can also be separately written into more user-friendly formats (for example, we might insert a row into a `chain_analyses` table with columns for each part for easy querying).

Additionally, we can incorporate vectorized long-term memory. Using `langgraph.checkpoint.postgres` along with `langchain_postgres` (as mentioned in Sajith K's article) allows storing conversational embeddings and doing semantic search . For instance, if the user has done multiple chain analyses, we can embed each summary and later, if the user's new situation is similar to a past one, automatically bring up the past lessons ("I recall you faced something similar on 2025-04-10 with an argument at work; you learned that lack of sleep was a big vulnerability then."). This adds a layer of personalization and context reuse. The LangGraph `PostgresStore` (for long-term memories) could be utilized to save important facts or summaries separately from the main conversation log .

To summarize, the LangGraph agent is implemented as a multi-step state machine that aligns with the chain analysis steps. It uses persistent state to carry information through the conversation and across sessions, enabled by a Postgres backend. Next, we discuss how each node will interact with the user through prompts, especially given the sensitive nature of therapy-related dialogue.

## Prompting Strategy and Safety Considerations

Designing the prompts for this AI agent requires special care. We must balance guiding the user through the steps with maintaining an empathetic, non-judgmental stance. Furthermore, we have to ensure the AI handles sensitive mental health content safely and ethically.

### Tone and Style

The system prompt given to the LLM will set a therapeutic tone. For instance, the assistant will be instructed to be kind, patient, and encouraging, somewhat like a therapist or a supportive coach. It should avoid sounding clinical or robotic. Each question should validate the user's feelings and use the user's own words when possible (e.g., "You mentioned [user's words]; let's explore that further…"). The agent will incorporate reflective listening: after a user shares something difficult (like a vulnerability or a consequence), the assistant may briefly acknowledge it ("I hear that you hadn't slept, which makes everything harder. That makes sense.") before moving on. This aligns with good counseling practices and helps the user feel understood.

### Clarity and Guidance

The prompts at each step should be clear about what information is needed, possibly giving examples. For example, when asking for vulnerability factors, we might include the categories explicitly in the question to jog the user's memory (physical, emotional, etc.). However, we should avoid overwhelming the user by asking about all categories at once – hence the plan to ask one by one. The assistant might say: "Let's consider a few areas that can make someone vulnerable: physical health, emotional state, substances, stress, or recent events. We'll go one by one." This heads-up prepares the user and ties into the DBT psychoeducation that these are common areas to check.

### Handling Sensitive Content

Users doing a chain analysis might divulge very sensitive information (e.g., self-harm, suicidal thoughts, trauma triggers, etc.). The AI must handle these disclosures carefully. Here are some measures we'll implement:

*   **Content Moderation and Emergency Protocols:** We will incorporate checks for signs of crisis. For example, if at any point the user's message indicates suicidal intent ("I actually don't want to live anymore") or intent to harm others, the agent should override the normal flow and respond with an appropriate urgent message. This could be done via a secondary classification model or by the LLM itself with a specialized prompt at each turn (e.g., an invisible "safety check" prompt asking if the last user message indicates self-harm or harm to others). Anthropic's principles and recent commentary (e.g., Bill Gates noting AI should detect emergencies) support this approach . In such cases, the assistant will not continue with chain analysis, but instead provide a compassionate message urging the user to seek immediate help (such as contacting a crisis hotline or a mental health professional), and perhaps even refusing to continue the exercise until safety is addressed. We will design the system prompt to prioritize user safety above completing the task.
*   **Validation and Non-Judgment:** The assistant will use validation strategies at each step. For example, if a user reveals a vulnerability like substance use or a prior self-harm attempt that day, the agent responds without judgment, normalizing that vulnerabilities happen: "It sounds like you had been drinking, which can really lower our guard. That's an important factor to note, and it's good that you recognize it." This helps reduce shame and encourages openness. We will include in the prompt instructions something like: "Always respond with understanding. Do not scold or shame the user for any behavior disclosed. Use phrases that validate their experience (e.g., 'it makes sense that…', 'it's understandable that…')."
*   **User Autonomy and Consent:** The agent should make it clear that the user can skip questions or take a break. We can build a mechanism where if a user says "I don't want to talk about that" for a particular vulnerability factor, the agent will respect that and move on. The state machine can have conditional transitions or flags for skipped steps. In practice, the LLM can decide to move to the next category if the user declines to answer one. We will incorporate that flexibility: e.g., if the user answers with something like "I'd rather not discuss my substance use," the agent will respond gently ("Okay, we can skip that topic if you prefer. How about other factors…") and proceed. The conversation design should never force an answer.
*   **Privacy and Boundaries:** The assistant will remind the user that their information is kept private and is being used to help them reflect (of course, subject to the system's actual privacy policy). If the user asks what will be done with their data, the assistant can explain in simple terms that it's stored securely to help remember context (we should predefine a helpful answer for this scenario, as it touches on trust).
*   **No Diagnostic or Prescriptive Behavior:** The agent's role is to guide the user's own analysis, not to diagnose or give authoritative advice beyond the DBT framework. We will direct the LLM (via system prompt) to avoid statements that sound like formal mental health diagnoses or any "you should do X" directives. Instead, it should ask questions and, when offering suggestions, frame them as possibilities: "Perhaps using a distraction skill (like going for a walk) at that point could have helped – what do you think?" This keeps the user in charge of their solutions, an important empowerment principle in therapy.

### Anthropic's Agent Principles

The design follows Anthropic's guidelines for building effective agents in several ways:

*   **Minimize unnecessary complexity:** We use straightforward prompt chaining and state control instead of overly complex multi-agent setups. The focus is on the core task. The entire flow could arguably be done with a single LLM prompt in theory, but by structuring it, we reduce cognitive load on the model at each step, which should improve accuracy and relevance of its questions.
*   **Maintain control and reliability:** By using LangGraph's orchestrated workflow, we ensure the agent doesn't skip steps or go off track. Each state acts as a checkpoint to verify that the necessary information was collected (we can even include simple programmatic checks or "gates" – e.g., if the user's description of the behavior lacks certain detail, the agent can ask one follow-up before proceeding, akin to the gate example in LangGraph docs). This aligns with the idea of keeping agents controllable and not entirely open-ended .
*   **Intermediate reflections:** We can leverage the LLM's ability to summarize or reformulate the user's input at times (like repeating back the chain in brief). This not only helps the user feel heard but also serves as a mild verification that the model correctly understood the user's message. If the summary is incorrect, the user can correct it, preventing misunderstandings from propagating.

### Prompt Format Example

In terms of prompt format, we will likely use a combination of system + user + assistant messages (as is standard in ChatML or similar). For example, a system prompt might contain:

```text
You are a DBT Chain Analysis assistant. You will guide the user through a series of steps to analyze a problematic behavior. Always be empathetic and use the principles of DBT. The steps are: 1) Problem behavior, 2) Prompting event, 3) Vulnerabilities (physical, emotional, substance, environment, prior actions), 4) Chain of events, 5) Consequences, 6) Solutions and prevention. Only move to the next step after the current one is fully discussed. Ask one question at a time. Validate the user's feelings. If the user seems distressed or mentions self-harm intention, respond with care and encourage seeking help (override the normal flow if it's an emergency). …
```

This system message sets the context. Each user message will be appended, and the assistant's role is to generate the next question or response accordingly. We will fine-tune these instructions during development with testing to ensure the assistant reliably follows the structure without excessive prompt length.

## Integration with Chainlit UI

### Chainlit User Interface

Chainlit will serve as the front-end, allowing the user to chat with the agent. The UI should be clean and focused – likely a simple chat bubble interface is enough, but we can enhance it with Chainlit's UI elements if needed. For instance, Chainlit allows adding rich components or forms, but in this case, the conversational format is primary. We might use the following Chainlit features in our implementation:

*   **Session Management:** Chainlit can maintain `user_session` data across the conversation. We will store references like the LangGraph graph object or the `thread_id` for the current session using `cl.user_session.set(...)` at session start. This way, each new message can retrieve the existing agent state machine instead of creating a new one.
*   **Message Streaming:** We'll utilize Chainlit's support for streaming tokens from the LLM's response. This means as the AI is formulating a possibly long prompt or summary, the user sees it in real time, improving interactivity. This is especially useful if the AI is enumerating a summary or multiple suggestions in the "Solutions" step, which could be a longer answer.
*   **Elements for Visualization:** If we choose, we might add a diagram or list output at the end. For example, after completing the chain, we could programmatically create a simple bullet list of the identified links or a table of vulnerabilities and solutions. Chainlit supports sending Markdown content (which we can just include in the assistant message). Alternatively, we could generate a graph (maybe similar to the chain diagram) on the fly using a library and send it as an image. However, this may not be necessary; a textual summary is often sufficient and more privacy-conscious (since graphical representation of someone's personal events might not add much value beyond text).
*   **Logging and Monitoring:** Chainlit, during development, will show the intermediate reasoning (especially if we use `cl.LangchainCallbackHandler()` to stream tool usage or prompts). In production, we will likely disable or filter out the intermediate steps from the user's view. However, we can keep them in server logs for debugging. Chainlit's debugging features (like an inspector for prompts) can help refine the prompt flow. Also, since we integrate a database, we have our own logging via Postgres.

### Chainlit and Database Integration

We have two layers of persistence – LangGraph's checkpoint system and possibly manual logging. The checkpoint (state) is saved in Postgres as described. But we may also implement a more straightforward logging of each turn in a separate table for analytics or compliance. This could be done in the Chainlit message handler: e.g., after receiving `message.content` from the user, insert a row into `Messages` table (with `user_id`, `session_id`, `role="user"`, `content`, `timestamp`); after generating the assistant response, insert a similar row (`role="assistant"`). However, doing this might be redundant if the entire conversation state is already stored.

One reason to log messages individually anyway is to facilitate historical context retrieval: If later we want to feed past dialogues into a vector search or simply display a past conversation to the user, having messages in a standard format is useful. We can maintain foreign keys linking messages to a session, and sessions to a user. This provides flexibility in querying the data (e.g., finding all instances of a certain behavior or trigger the user has discussed).

To connect Chainlit with Postgres, we will use an ORM (like SQLAlchemy) or direct queries via `psycopg` to perform these inserts/selects. Since Chainlit's `on_chat_start` and `on_message` are async functions, any DB calls will be awaited accordingly.

### Example Chainlit Handlers

Pseudocode below demonstrates how the Chainlit event handlers might be implemented to wire everything up:

```python
import chainlit as cl
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
# Assume other necessary imports like LLM, state definition, etc.

# Assume we have defined the StateGraph for chain analysis and a function to create it
# chain_analysis_graph_definition = create_chain_analysis_graph_definition() # Returns a StateGraph definition
# Connect the graph to Postgres for persistence
# pg_saver = PostgresSaver.from_conn_string("postgresql://user:pass@host:port/db") # Example connection
# compiled_graph = chain_analysis_graph_definition.compile(checkpointer=pg_saver)

# Utility functions (placeholders)
def generate_guest_id(): return "guest_" + str(uuid.uuid4())
def fetch_last_summary(user_id): return None # Placeholder DB call
def save_message_to_db(session_id, text, role): pass # Placeholder DB call
def extract_last_assistant_message(state): 
    # Logic depends on how messages are stored in state
    # Example: return state['messages'][-1].content if state['messages'][-1].type == 'ai' else ""
    return state.get("assistant_msg", "") # Simplified from previous example
async def stream_tokens(message_content): 
    # Placeholder for actual token streaming logic
    for char in message_content: yield char 

@cl.on_chat_start
async def on_chat_start():
    # Setup the graph instance for the session
    pg_saver = PostgresSaver.from_conn_string("YOUR_PG_CONN_STRING") # Get connection string securely
    # Assume chain_analysis_graph_definition is defined globally or loaded
    compiled_graph = chain_analysis_graph_definition.compile(checkpointer=pg_saver)
    cl.user_session.set('compiled_graph', compiled_graph)

    # Identify user/session
    user_id = cl.user_session.get('user_id') # Assume user ID is managed elsewhere or use guest ID
    if not user_id:
        user_id = generate_guest_id()
        cl.user_session.set('user_id', user_id)
        
    session_id = f"{user_id}_{str(uuid.uuid4())}" # Create a unique thread ID for LangGraph
    cl.user_session.set('session_id', session_id)
    
    # Perhaps load last session summary for user (if any) to personalize
    last_summary = fetch_last_summary(user_id) # Implement DB query
    if last_summary:
        intro_msg = (f"(The assistant recalls some context from your last session: " 
                     f"{last_summary})\n\nReady to begin a new chain analysis when you are.")
        await cl.Message(content=intro_msg, author="assistant").send()
    else:
        # Send a greeting or prompt user to start
        await cl.Message(content="Hello, I'm here to help you do a Chain Analysis. Let me know what behavior you want to analyze today.", author="assistant").send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get('session_id')
    compiled_graph = cl.user_session.get('compiled_graph')
    user_text = message.content

    # Log user message to DB (optional, could rely on checkpoint instead)
    save_message_to_db(session_id, user_text, role="user") # Implement DB save

    # Prepare input state for the agent. 
    # Depends on graph state definition. If using message list:
    # input_state = {"messages": [HumanMessage(content=user_text)]}
    # If just passing text to a node expecting it:
    input_state = {"user_input": user_text} # Example, adjust based on graph node design

    config = {"configurable": {"thread_id": session_id}}
    
    # Invoke LangGraph state machine with the new user input
    # Use await graph.ainvoke(...) for async operation
    result_state = await compiled_graph.ainvoke(input_state, config=config) 
    
    # The result_state will contain updated state including the assistant's latest message(s)
    # Extract the assistant's response (logic depends on state structure)
    assistant_msg = extract_last_assistant_message(result_state) # Implement extraction logic

    # Send assistant response to user (streaming)
    resp = cl.Message(content="")
    async for token in stream_tokens(assistant_msg): # Implement async streaming
        await resp.stream_token(token)
    await resp.send()
    
    # Log assistant message to DB
    save_message_to_db(session_id, assistant_msg, role="assistant") # Implement DB save
```

In this pseudocode, `create_chain_analysis_graph_definition()` would build the `StateGraph` with all the nodes as described. The `thread_id` is how LangGraph's `PostgresSaver` links to the checkpoint state . Every time we invoke the graph with the same `thread_id`, it will load the last saved state and then append the new input (using appropriate state annotations like `add_messages` if the state includes a message list ). The assistant's response is then generated accordingly. We stream it out via Chainlit's mechanisms. This illustrates how Chainlit's backend coordinates between the UI, the LangGraph agent, and the Postgres database.

### Chainlit UI/UX Enhancements

We can leverage some UI features to improve user experience:

*   Use Markdown formatting in the assistant messages for readability. For instance, when summarizing the chain, the agent can use bullet points or numbered lists to enumerate links or solutions, making it clearer. (Chainlit will render Markdown in the chat.)
*   Possibly use emojis or slight stylistic elements sparingly to convey empathy (like "I'm really sorry you went through that 😞" if appropriate). However, this should be done carefully and professionally.
*   If the conversation becomes long, the assistant might occasionally summarize or confirm ("So far, we have…") which helps the user recall previous parts without scrolling. This can be done by the LLM proactively as part of the prompt for each new step (we instruct it to reference what's already gathered).
*   At the end, ask the user if they want to save or email the summary of the chain analysis (if such features are in scope). Chainlit could allow downloading the conversation. But given privacy concerns, maybe just inform the user they can copy the chat text if they want to keep it.

## Database Schema and Historical Context Usage

A proper database design will support the features above. We propose the following schema outline in PostgreSQL to persist data:

*   **`Users` table:** (`user_id`, `name`, `email`, etc.) – if the app has user accounts. If not, we may skip this and use session tokens. But having a user concept is useful for long-term personalization.
*   **`Sessions` table:** (`session_id` (PK), `user_id` (FK to Users), `start_time`, `end_time`, `type`) – where `type` might indicate this is a "chain_analysis" session (versus other conversations if the app grows). This table tracks each distinct chain analysis conversation. Corresponds to LangGraph's `thread_id`.
*   **`Messages` table:** (`msg_id` (PK), `session_id` (FK to Sessions), `role`, `content`, `timestamp`) – logs each utterance. `role` is either "user" or "assistant" (or system if we log system prompts too). `content` stores the text. For long messages, this can be `TEXT` type. We might also store `msg_embedding` if using pgvector for semantic search on message content (to find similar phrasing etc., though not a priority).
*   **`ChainAnalysisResults` table:** (`result_id` (PK), `session_id` (FK to Sessions), `problem_behavior`, `prompting_event`, `vulnerabilities_json`, `chain_json`, `consequences`, `solutions_json`, `summary_text`, `created_at`) – this would store the outcome in structured form for quick reference. `vulnerabilities_json` can be a JSON blob of all factors, `chain_json` could be the list of links as JSON, `solutions_json` similarly. `summary_text` is the human-readable summary. This is optional but useful for generating reports or reviewing progress over time. It basically duplicates what's in the final LangGraph state but makes it easier to query without loading the entire message history or checkpoint.

LangGraph's `PostgresSaver` by default creates a `checkpoints` table (as shown in the medium article) with columns: `thread_id`, `checkpoint_id`, `parent_checkpoint_id`, `checkpoint` (probably a JSON or binary field), `metadata` . We can rely on that for restoring state, but for our own querying and display, we'll use the above tables. The `thread_id` we use in LangGraph will correspond to our `session_id`. Each time the graph state updates, a new checkpoint row is added with an incremented `checkpoint_id`. We could even reconstruct the conversation from these (since messages might be part of state), but it's often easier to just log messages separately as they come.

### Referencing History

When a user returns for a new session, the assistant can tap into the database to see what previous sessions exist for that user. We will implement logic such as:

*   If a user explicitly asks something like "Have we talked about X before?" or "Remind me what I learned last time," the agent can perform a look-up. For example, do a semantic search in the `ChainAnalysisResults.summary_text` for similar topics, or just fetch the last session's summary from the `ChainAnalysisResults` table for that `user_id`.
*   Even if not asked, the agent might proactively bring relevant context. For instance, if this is a second chain analysis and the problem behavior is similar to last time, the agent might say "I recall last time you analyzed a similar behavior where lack of sleep was a vulnerability. Let's see if that's a factor again or if something changed." This kind of continuity can make the experience feel personalized. We'll only do this if we're confident it's helpful and not seen as invasive. The user could always respond "actually that was different" and the agent would adjust.
*   The fidelity of recall can be tuned. We could have the assistant rely on the LLM's memory by injecting a brief summary of the last session into the system prompt for the new session. Or, use it as a separate user message at the start: "(Context from last session: …)" which the assistant then has as part of the dialogue. This was hinted in the pseudocode as `last_summary = fetch_last_summary(user_id)`. We will generate that summary either by retrieving a stored one (from `ChainAnalysisResults`) or by programmatically summarizing the final state of the last session with another LLM call offline when the session ends.

By maintaining history, we also enable progress tracking: The database could be used to chart if certain vulnerabilities (say "anger" or "sleep issues") keep coming up for the user. While not required in this plan, it opens the door to analytics that could be shared with the user or their clinician with consent, thus adding value beyond each individual chat.

### Security and Privacy in Data

With sensitive psychological data in Postgres, securing the database is paramount. We will enforce encryption at rest, use proper authentication, and ensure that the Chainlit server only queries data for the authenticated user's sessions. If multi-user, each user should not access others' data. These are implementation details but worth noting in the plan for completeness.

## Example Dialogue Flow

To illustrate how all these pieces come together, here's a condensed example of a user interaction (without showing every system detail, but how we expect the conversation to flow):

*   **User:** Hi, I need help analyzing something I did yesterday.
*   **Assistant:** Sure. I'm here to help you do a Chain Analysis of a behavior. Let's start. What behavior would you like to focus on? *(Step 1 prompt)*
*   **User:** I drank again even though I'm trying to quit. I feel awful about it.
*   **Assistant:** I'm sorry you're feeling that way, but I'm glad you're here working on this. So the behavior we'll analyze is drinking alcohol after trying to quit. I've noted that. Now, can you tell me what set this off yesterday – what was the prompting event that happened before you started drinking? *(Step 2 prompt, with reflection)*
*   **User:** I had a fight with my partner last evening.
*   **Assistant:** Okay. The chain started with an argument with your partner. That's the prompting event. Next, we'll consider what might have made you extra vulnerable before that argument happened. I'll ask a few areas one by one. First, physically – how were you feeling yesterday? Any illness, exhaustion, hunger, or physical pain that day? *(Step 3, sub-question 1)*
*   **User:** Actually, I barely ate anything all day and I was really tired.
*   **Assistant:** Lack of food and fatigue can definitely increase vulnerability. I'm noting those. How about emotional state – were you already feeling upset or stressed before the fight? *(Step 3, sub-question 2)*
*   **User:** Yes, I was anxious all afternoon about a work deadline.
*   **Assistant:** Understood. Anxiety was high before it even happened. Any substance use earlier or medications? *(Step 3, sub-question 3)*
*   **User:** No, I was sober until I started drinking after the fight.
*   **Assistant:** Okay. Any other stressful events or things in the environment yesterday? *(Step 3, sub-question 4)*
*   **User:** Not really, just the usual stuff.
*   **Assistant:** And did you do anything earlier that day that left you feeling bad? For example, sometimes we might have had another conflict or we skipped something important. *(Step 3, sub-question 5)*
*   **User:** I had procrastinated my work, which is why I was anxious. So I guess my own actions added stress.
*   **Assistant:** That makes sense – procrastinating work increased your stress later. So, summarizing your vulnerabilities yesterday: very little eating, feeling very tired, high anxiety about work (partly due to procrastination). These factors set the stage. Now let's look at the chain of events after the argument started. What was the first thing that happened during or after the fight, emotionally or in your actions? *(Step 4 begin)*
*   **User:** During the fight I just exploded in anger, I yelled a lot. After that I felt guilty and really upset.
*   **Assistant:** So you got very angry and yelled. That's one link. Then you felt guilt and upset. What did you do next with those feelings?
*   **User:** I went to the kitchen and grabbed a beer.
*   **Assistant:** You started drinking beer. What was going through your mind at that moment, if you recall?
*   **User:** I thought, "I can't deal with this, I need a drink to numb out."
*   **Assistant:** Got it. Thought: can't deal, need to numb. Then you drank the beer. Did it stop at one beer, or…?
*   **User:** I kept going and got drunk. That's the behavior – I ended up drinking a lot.
*   **Assistant:** Okay, so that's the chain: Argument → Anger (yelling) → Guilt/Upset → Urge to numb (thought) → Drank one beer → which led to many beers (drunkenness), which is the target behavior we're analyzing. Now, let's talk about what happened after you started drinking (the consequences). How did you feel afterward, and how did your partner respond? *(Step 5 prompt)*
*   **User:** My partner was even more upset and left the house. I felt ashamed and sick. This morning I had a horrible headache and even more guilt.
*   **Assistant:** Consequences noted: your partner left upset, you felt shame right after, and today you have a hangover and guilt. These are painful outcomes. Now, let's think of some ways you could handle things differently next time (or even last night if you had a redo). *(Step 6 intro)* First, looking at the chain, where do you think you could have done something that would have led to a better result?
*   **User:** Maybe after the fight, when I was feeling guilty, I could have gone for a walk instead of drinking. Also, I could have tried to talk it out or apologize rather than just giving up.
*   **Assistant:** Those are great ideas – take a walk to cool down, or attempt repair with your partner. Those could have prevented the drinking. Even earlier, perhaps if you recognized "I can't deal", you might have… maybe texted a friend or used a relaxation technique rather than reaching for beer. Does that sound possible?
*   **User:** Yeah, I have a friend I could have called. I also have some breathing exercises I learned; I just didn't think of them.
*   **Assistant:** Right, it's hard in the moment, but good to note. Now, looking at the vulnerabilities, some things might be preventable in the future. For example, eating properly and not skipping meals could help, and addressing the procrastination so the work anxiety isn't so high. Think about sleep, food, managing stress – anything you plan to do differently on that front?
*   **User:** I will try to not skip lunch, no matter how busy I get. And I'll set a schedule to tackle work so I don't leave it all last minute.
*   **Assistant:** Excellent. So to recap solutions: During a future conflict – try taking a walk or using a calming skill or calling a friend instead of drinking. Before conflicts happen – keep yourself less vulnerable by eating and resting, and by managing your work tasks proactively. These are solid plans.
*   **User:** Thank you. This really helps.
*   **Assistant:** You're very welcome. You've done a thorough analysis of what happened. To summarize quickly: Yesterday you were vulnerable because you were exhausted and anxious. The prompting event was a fight with your partner. That led you to get very angry, feel guilty, and then turn to alcohol to cope. The consequences were harmful – your partner was upset and you felt worse later. Next time, you have ideas to handle it differently, like walking away to cool off, reaching out to a friend, and reducing your baseline stress by not skipping meals and managing work better. Great work identifying all this! Keep this in mind moving forward. And remember, if you find yourself in a similar situation, you can use these insights to make different choices. Would you like a copy of this summary for later reference?
*   **User:** Yes, please.
*   **Assistant:** Sure! *(The assistant provides a formatted summary.)* And if you ever want to do this again or talk through things, I'm here. Take care and good luck – you've got this.

*(End of session – the assistant saves the summary and perhaps offers a final encouraging closing.)*

This dialogue demonstrates how the agent follows the chain analysis steps in order, how it asks about vulnerabilities and links, and how it responds to the user in a supportive manner. The agent also clearly distinguishes each stage (the user can tell what they are focusing on at a given moment), which helps with the transparency of the process.

## Conclusion

In summary, the proposed AI Chain Analysis Assistant will be built as a guided conversational agent that systematically helps users dissect a problematic behavior episode. By using LangGraph to encode the DBT chain analysis procedure as a sequence of states, we ensure that the AI covers all essential steps (problem, trigger, vulnerabilities, chain, consequences, solutions) with reliability. The integration with Chainlit provides a user-friendly chat interface and allows real-time interaction, while PostgreSQL persistence gives the agent long-term memory, enabling continuity and personalization across sessions . Throughout the interaction, the system adheres to safety and ethical standards, prompting users about sensitive topics in a thoughtful way and incorporating fail-safes for crisis situations.

This implementation plan also aligns with principles from Anthropic's agent framework: preferring a clear workflow over unpredictable autonomy for this use case, but still leveraging the LLM's natural language prowess for empathy and flexibility. We have outlined how each component will be structured (from state graph nodes to database tables), and provided examples of prompts and interactions.

Moving forward, development will involve:

*   Setting up the Python environment with LangGraph, Chainlit, and necessary LLM APIs.
*   Implementing the `StateGraph` nodes and transitions as per the design, and writing extensive unit tests or simulations for each step (e.g., ensuring that skipping a vulnerability still leads to the next step correctly, etc.).
*   Iteratively refining the system prompt and few-shot examples (if needed) to calibrate the assistant's tone and adherence to the sequence.
*   Using Chainlit's interface to conduct user testing sessions and collect feedback from mental health professionals to ensure the process is therapeutically sound.
*   Hardening the system with the discussed safety checks (this might involve additional classification models or rules integrated via LangGraph tool calls or simply logic around the user's messages).
*   Deploying the system, likely as a web app behind authentication if used in a real client context, and monitoring it with logs and possible human-in-the-loop review initially (especially to verify it's making safe choices on tricky inputs).

By following this plan, we aim to create an AI assistant that not only helps users perform chain analysis conveniently but does so in a way that is personal, safe, and effective. The result will be a powerful tool for clients to gain insight into their behaviors and patterns, augmenting the therapeutic process with on-demand AI support.

## References

*   DBT Self-Help – Target Behaviors & Behavior Chain Analysis: Provided the step-by-step structure of a behavior chain analysis, including the key focus on vulnerability factors.
*   Anthropic – Building Effective Agents: Informed the agent design philosophy, emphasizing simple, structured workflows for tasks like ours .
*   LangGraph Documentation and Tutorials: Guided the use of LangGraph for creating stateful LLM workflows, including persistent memory with Postgres and streaming responses .
*   LangChain/LangGraph + Chainlit Integration Guide: Instructed how to integrate LangGraph agents with the Chainlit UI event loop, which influenced our Chainlit handler design.
*   LangChain Blog (Mental Health as State Machine): Demonstrated the viability of modeling therapy conversations as state machines and inspired parts of our FSM approach, reinforcing the balance between structure and flexibility in a therapeutic context.
