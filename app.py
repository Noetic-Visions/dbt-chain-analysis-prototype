import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

from dbt.graphs.assistant import assistant_graph


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}

    # Use simple config without callbacks to avoid LangSmith tracing errors
    config_with_thread = RunnableConfig(**config)

    final_answer = cl.Message(content="")

    for msg, metadata in assistant_graph.stream(
        {"messages": [HumanMessage(content=message.content)]},
        stream_mode="messages",
        config=config_with_thread,
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "dialogue"
        ):
            # Handle both string content and message chunks
            if isinstance(msg, AIMessageChunk):
                content = str(msg.content) if msg.content else ""
            else:
                content = str(msg.content) if msg.content else ""

            if content:
                await final_answer.stream_token(content)

    await final_answer.send()
