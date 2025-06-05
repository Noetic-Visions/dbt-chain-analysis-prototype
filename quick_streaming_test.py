#!/usr/bin/env python3
"""Quick streaming test with a simple query."""

from langchain_core.messages import HumanMessage

from simple_langgraph_example import create_comprehensive_agent


def quick_streaming_test():
    """Quick test to see streaming in action."""
    print("🌊 Quick Streaming Test")
    print("=" * 40)

    agent = create_comprehensive_agent()

    print("🔍 Asking for just file stats (should be quick)...\n")

    for chunk in agent.stream(
        {
            "messages": [
                HumanMessage(
                    content="Get the file statistics for app.py and tell me what you think it might be used for."
                )
            ]
        }
    ):
        # Handle different types of chunks
        if "agent" in chunk and chunk["agent"].get("messages"):
            message = chunk["agent"]["messages"][-1]
            if hasattr(message, "content"):
                print(message.content, end="", flush=True)
        elif "tools" in chunk:
            tool_messages = chunk["tools"].get("messages", [])
            for tool_msg in tool_messages:
                if hasattr(tool_msg, "content"):
                    print(f"\n🔧 {tool_msg.content}\n", flush=True)

    print("\n\n✅ Quick test complete!")


if __name__ == "__main__":
    quick_streaming_test()
