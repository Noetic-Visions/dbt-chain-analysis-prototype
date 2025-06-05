#!/usr/bin/env python3
"""Quick test script to analyze a specific file with our comprehensive search tools - WITH STREAMING!"""

import sys

from langchain_core.messages import HumanMessage

from simple_langgraph_example import create_comprehensive_agent


def analyze_file_streaming(file_path: str):
    """Analyze a file using our comprehensive search tools with streaming output."""
    print(f"🔍 Analyzing {file_path}")
    print("=" * 60)
    print("🌊 Streaming response...\n")

    # Create the agent
    agent = create_comprehensive_agent()

    # Comprehensive analysis query
    messages = [
        HumanMessage(
            content=f"""
        Please do a comprehensive analysis of {file_path}:

        1. First, get the file statistics
        2. Search for class definitions (look for 'class ')
        3. Search for function definitions (look for 'def ')
        4. Search for import statements (look for 'import ' and 'from ')
        5. If you find any interesting classes or functions, extract a few lines around them for context

        Based on your analysis, tell me:
        - What kind of file this appears to be
        - What are the main classes/functions
        - What external libraries it uses
        - Overall purpose/functionality
        """
        )
    ]

    # Stream the response
    try:
        for chunk in agent.stream({"messages": messages}):
            # Handle different types of chunks
            if "agent" in chunk and chunk["agent"].get("messages"):
                # Agent response chunk
                message = chunk["agent"]["messages"][-1]
                if hasattr(message, "content"):
                    print(message.content, end="", flush=True)
            elif "tools" in chunk:
                # Tool execution chunk
                tool_messages = chunk["tools"].get("messages", [])
                for tool_msg in tool_messages:
                    if hasattr(tool_msg, "content"):
                        print(f"\n🔧 Tool: {tool_msg.content}\n", flush=True)

        print("\n\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Streaming failed, falling back to regular invoke: {e}")
        # Fallback to regular invoke
        result = agent.invoke({"messages": messages})
        print(result["messages"][-1].content)


def analyze_file_regular(file_path: str):
    """Regular non-streaming analysis for comparison."""
    print(f"🔍 Analyzing {file_path} (Regular Mode)")
    print("=" * 60)

    # Create the agent
    agent = create_comprehensive_agent()

    # Comprehensive analysis query
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""
            Please do a comprehensive analysis of {file_path}:

            1. First, get the file statistics
            2. Search for class definitions (look for 'class ')
            3. Search for function definitions (look for 'def ')
            4. Search for import statements (look for 'import ' and 'from ')
            5. If you find any interesting classes or functions, extract a few lines around them for context

            Based on your analysis, tell me:
            - What kind of file this appears to be
            - What are the main classes/functions
            - What external libraries it uses
            - Overall purpose/functionality
            """
                )
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    import sys

    # Choose mode based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "regular":
        print("🐌 Running in regular (non-streaming) mode\n")
        analyze_file_regular("dbt_supervisor_main.py")
    else:
        print("🌊 Running in streaming mode\n")
        analyze_file_streaming("dbt_supervisor_main.py")
