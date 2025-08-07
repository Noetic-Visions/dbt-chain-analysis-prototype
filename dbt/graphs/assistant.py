import re
from typing import Annotated

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, SecretStr

from dbt.loggers import get_logger

logger = get_logger(__name__)


def clean_qwen_output(ai_message) -> str:
    """Clean Qwen output by removing <think> tags."""
    return re.sub(
        r"<think>.*?</think>", "", ai_message.content, flags=re.DOTALL
    ).strip()


model = (
    ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-30b-a3b",
        api_key=SecretStr("123"),
        temperature=0.6,
        top_p=0.95,
        model_kwargs={"extra_body": {"top_k": 40}},
    )
    | clean_qwen_output
)


class AssistantState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str = Field(default="")


def call_model(state: AssistantState):
    # Get summary if it exists
    summary = state.summary

    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state.messages

    else:
        messages = state.messages

    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: AssistantState):
    # First, we get any existing summary
    summary = state.summary

    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state.messages + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 3 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-3] if m.id]
    return {"summary": response, "messages": delete_messages}


def should_continue(state: AssistantState):
    """Return the next node to execute."""

    messages = state.messages

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        logger.info("Summarizing conversation")
        return "summarize_conversation"

    # Otherwise we can just end
    return END


def build_graph():
    memory = MemorySaver()
    graph = StateGraph(AssistantState)

    graph.add_node("dialogue", call_model)
    graph.add_node(summarize_conversation)

    graph.add_edge(START, "dialogue")
    graph.add_conditional_edges("dialogue", should_continue)
    graph.add_edge("summarize_conversation", END)

    workflow = graph.compile(checkpointer=memory)

    return workflow


assistant_graph = build_graph()
