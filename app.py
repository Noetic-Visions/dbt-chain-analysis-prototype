import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

from dbt.graphs.assistant import assistant_graph


class ThinkingParser:
    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
        self.thinking_step = None

    async def process_chunk(self, chunk: str, final_answer: cl.Message):
        """Process a chunk of text, handling <think> blocks separately"""
        self.buffer += chunk

        while True:
            if not self.in_thinking:
                # Look for opening think tag
                think_start = self.buffer.find("<think>")
                if think_start == -1:
                    # No think tag found, stream all current buffer to final answer
                    if self.buffer:
                        await final_answer.stream_token(self.buffer)
                        self.buffer = ""
                    break
                else:
                    # Stream content before think tag to final answer
                    if think_start > 0:
                        await final_answer.stream_token(self.buffer[:think_start])

                    # Remove processed content and enter thinking mode
                    self.buffer = self.buffer[think_start + 7 :]  # 7 = len("<think>")
                    self.in_thinking = True

                    # Create thinking step
                    self.thinking_step = cl.Step(name="Thinking", type="thinking")
                    self.thinking_step.input = "Model is thinking..."
                    await self.thinking_step.send()
            else:
                # Look for closing think tag
                think_end = self.buffer.find("</think>")
                if think_end == -1:
                    # No closing tag yet, stream current buffer to thinking step
                    if self.buffer:
                        await self.thinking_step.stream_token(self.buffer)
                        self.buffer = ""
                    break
                else:
                    # Stream thinking content to step
                    if think_end > 0:
                        await self.thinking_step.stream_token(self.buffer[:think_end])

                    # Close thinking step
                    await self.thinking_step.update()

                    # Remove processed content and exit thinking mode
                    self.buffer = self.buffer[think_end + 8 :]  # 8 = len("</think>")
                    self.in_thinking = False
                    self.thinking_step = None


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}

    # Use simple config without callbacks to avoid LangSmith tracing errors
    config_with_thread = RunnableConfig(**config)

    final_answer = cl.Message(content="")
    parser = ThinkingParser()

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
                await parser.process_chunk(content, final_answer)

    await final_answer.send()
