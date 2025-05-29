import time

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

from dbt.graphs.assistant import assistant_graph


async def process_stream_with_thinking(stream, thinking_step, final_answer, start_time):
    """Process the LangGraph stream and handle thinking content within the step context"""
    thinking = False
    has_thinking_content = False

    for msg, metadata in stream:
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
                # Process content character by character to handle tags properly
                i = 0
                while i < len(content):
                    # Check for <think> tag
                    if content[i : i + 7] == "<think>":
                        thinking = True
                        has_thinking_content = True
                        i += 7
                        continue

                    # Check for </think> tag
                    if content[i : i + 8] == "</think>":
                        thinking = False
                        thought_for = round(time.time() - start_time)
                        thinking_step.name = f"Thought for {thought_for}s"
                        await thinking_step.update()
                        i += 8
                        continue

                    # Stream single character
                    char = content[i]
                    if thinking:
                        await thinking_step.stream_token(char)
                    else:
                        await final_answer.stream_token(char)

                    i += 1

    return has_thinking_content


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    config_with_thread = RunnableConfig(**config)

    start = time.time()

    # Create the stream
    stream = assistant_graph.stream(
        {"messages": [HumanMessage(content=message.content)]},
        stream_mode="messages",
        config=config_with_thread,
    )

    # Handle thinking content within the step context
    async with cl.Step(name="Thinking", default_open=True) as thinking_step:
        final_answer = cl.Message(content="")

        has_thinking = await process_stream_with_thinking(
            stream, thinking_step, final_answer, start
        )

        # If no thinking content was found, update step to indicate that
        if not has_thinking:
            thinking_step.name = "No thinking required"
            await thinking_step.update()

    await final_answer.send()
