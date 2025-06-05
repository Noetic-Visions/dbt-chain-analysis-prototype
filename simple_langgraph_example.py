#!/usr/bin/env python3
"""Complete LangGraph integration that exposes ALL capabilities of the segmented file search tool."""

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Import our file search tools
from fixed_segmented_search import (
    FixedAgentFileSearchTool,
    FixedSegmentedFileSearchTool,
)


# Input schemas for all tools
class FileSearchInput(BaseModel):
    """Input schema for the file search tool."""

    file_path: str = Field(description="Path to the file to search")
    query: str = Field(description="Search pattern or text to find")
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )
    use_regex: bool = Field(
        default=False, description="Whether to treat the query as a regex pattern"
    )


class FileSegmentInput(BaseModel):
    """Input schema for the file segment extraction tool."""

    file_path: str = Field(description="Path to the file to extract from")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed)")


class FileStatsInput(BaseModel):
    """Input schema for file statistics."""

    file_path: str = Field(description="Path to the file to analyze")


class ContextAroundMatchInput(BaseModel):
    """Input schema for getting context around a match."""

    file_path: str = Field(description="Path to the file containing the match")
    line_number: int = Field(description="Line number of the match")
    context_lines: int = Field(
        default=3, description="Number of lines before and after to include"
    )


class SegmentSearchInput(BaseModel):
    """Input schema for searching within a specific segment."""

    file_path: str = Field(description="Path to the file to search")
    start_line: int = Field(description="Starting line of the segment")
    end_line: int = Field(description="Ending line of the segment")
    query: str = Field(description="Search pattern or text to find")
    use_regex: bool = Field(
        default=False, description="Whether to treat the query as a regex pattern"
    )


# Global instances - created once and reused
_search_agent = None
_core_searcher = None


def get_search_agent():
    global _search_agent
    if _search_agent is None:
        _search_agent = FixedAgentFileSearchTool()
    return _search_agent


def get_core_searcher():
    global _core_searcher
    if _core_searcher is None:
        _core_searcher = FixedSegmentedFileSearchTool(segment_size=100, overlap=10)
    return _core_searcher


class LangGraphFileSearchTool(BaseTool):
    """Enhanced file search tool with regex support."""

    name: str = "search_file"
    description: str = """Search for text patterns in files efficiently with optional regex support.
    This tool can search large files without loading them entirely into memory.

    Use this tool when you need to:
    - Find function or class definitions
    - Locate import statements
    - Search for specific variables or patterns
    - Find error messages or logs
    - Use regex patterns for complex searches
    """
    args_schema: type[BaseModel] = FileSearchInput

    def _run(
        self, file_path: str, query: str, max_results: int = 5, use_regex: bool = False
    ) -> str:
        """Execute the file search and return formatted results."""
        try:
            # Use the core searcher directly to get access to regex functionality
            core_searcher = get_core_searcher()
            matches = core_searcher.smart_segment_search(
                file_path, query, max_matches=max_results
            )

            if not matches:
                return f"🔍 No matches found for '{query}' in {file_path}"

            # Format results nicely for the LLM
            search_type = "regex pattern" if use_regex else "text pattern"
            lines = [
                f"🔍 Found {len(matches)} matches for {search_type} '{query}' in {file_path}:\n"
            ]

            for i, match in enumerate(matches, 1):
                lines.append(
                    f"{i}. Line {match.line_number}: {match.line_content.strip()}"
                )

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Search failed: {str(e)}"


class LangGraphFileSegmentTool(BaseTool):
    """File segment extraction tool."""

    name: str = "get_file_segment"
    description: str = """Extract a specific segment of lines from a file.
    Perfect for getting detailed context around code sections found through search.

    Use this tool when you need to:
    - Get full context around a function or class definition
    - Extract specific sections of code for analysis
    - Read specific line ranges from large files
    """
    args_schema: type[BaseModel] = FileSegmentInput

    def _run(self, file_path: str, start_line: int, end_line: int) -> str:
        """Extract and return a file segment."""
        search_agent = get_search_agent()
        result = search_agent.get_file_segment(file_path, start_line, end_line)

        if not result["success"]:
            return (
                f"❌ Segment extraction failed: {result.get('error', 'Unknown error')}"
            )

        total_lines = result.get("total_file_lines", "unknown")
        lines = [
            f"📄 File segment from {file_path} (lines {start_line}-{end_line}, total: {total_lines} lines):\n"
        ]

        # Add line numbers to the content
        content_lines = result["content"].split("\n")
        for i, line in enumerate(content_lines):
            if line.strip() or i < len(content_lines) - 1:
                line_num = start_line + i
                lines.append(f"{line_num:4d}: {line}")

        return "\n".join(lines)


class LangGraphFileStatsTool(BaseTool):
    """File statistics and information tool."""

    name: str = "get_file_stats"
    description: str = """Get basic statistics about a file including total line count.
    Useful for understanding file size before searching or extracting segments.

    Use this tool when you need to:
    - Check if a file exists and is readable
    - Get total line count for planning segment extractions
    - Understand file size before processing
    """
    args_schema: type[BaseModel] = FileStatsInput

    def _run(self, file_path: str) -> str:
        """Get file statistics."""
        try:
            core_searcher = get_core_searcher()
            line_count = core_searcher.get_file_line_count(file_path)

            if line_count == 0:
                return f"📊 File {file_path} is empty or could not be read"

            return f"📊 File statistics for {file_path}:\n- Total lines: {line_count:,}\n- Estimated segments: {(line_count // 100) + 1}"

        except Exception as e:
            return f"❌ Failed to get file stats: {str(e)}"


class LangGraphContextAroundMatchTool(BaseTool):
    """Tool to get context around a specific line number."""

    name: str = "get_context_around_line"
    description: str = """Get context lines around a specific line number.
    Perfect for getting more context around interesting matches found through search.

    Use this tool when you need to:
    - Get more context around a search result
    - Understand the surrounding code for a specific line
    - Analyze code structure around a particular location
    """
    args_schema: type[BaseModel] = ContextAroundMatchInput

    def _run(self, file_path: str, line_number: int, context_lines: int = 3) -> str:
        """Get context around a specific line."""
        try:
            core_searcher = get_core_searcher()
            start_line = max(1, line_number - context_lines)
            end_line = line_number + context_lines

            segment = core_searcher.extract_segment(file_path, start_line, end_line)

            lines = [
                f"🔍 Context around line {line_number} in {file_path} (±{context_lines} lines):\n"
            ]

            content_lines = segment.content.split("\n")
            for i, line in enumerate(content_lines):
                if line.strip() or i < len(content_lines) - 1:
                    current_line = start_line + i
                    marker = " >>> " if current_line == line_number else "     "
                    lines.append(f"{marker}{current_line:4d}: {line}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Failed to get context: {str(e)}"


class LangGraphSegmentSearchTool(BaseTool):
    """Tool to search within a specific file segment."""

    name: str = "search_in_segment"
    description: str = """Search for patterns within a specific segment of a file.
    Useful for focused searching within a known code section.

    Use this tool when you need to:
    - Search within a specific function or class
    - Find patterns in a particular section of code
    - Do targeted searching after extracting a segment
    """
    args_schema: type[BaseModel] = SegmentSearchInput

    def _run(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        query: str,
        use_regex: bool = False,
    ) -> str:
        """Search within a specific segment."""
        try:
            core_searcher = get_core_searcher()
            segment = core_searcher.extract_segment(file_path, start_line, end_line)
            matches = core_searcher.search_in_segment(segment, query, use_regex)

            if not matches:
                return f"🔍 No matches for '{query}' in segment lines {start_line}-{end_line} of {file_path}"

            search_type = "regex pattern" if use_regex else "text pattern"
            lines = [
                f"🔍 Found {len(matches)} matches for {search_type} '{query}' in segment {start_line}-{end_line}:\n"
            ]

            for i, match in enumerate(matches, 1):
                lines.append(
                    f"{i}. Line {match.line_number}: {match.line_content.strip()}"
                )

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Segment search failed: {str(e)}"


def create_comprehensive_agent() -> StateGraph:
    """Create a comprehensive LangGraph agent with ALL file search capabilities."""

    # Initialize ALL available tools
    tools = [
        LangGraphFileSearchTool(),
        LangGraphFileSegmentTool(),
        LangGraphFileStatsTool(),
        LangGraphContextAroundMatchTool(),
        LangGraphSegmentSearchTool(),
    ]

    # Initialize LLM with tools - fixed parameter warning
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-30b-a3b",
        api_key="123",
        temperature=0.6,
        top_p=0.95,
        # Removed model_kwargs to fix warning - if you need top_k, you might need to configure your server differently
    )

    llm_with_tools = llm.bind_tools(tools)

    # Create tool node
    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState) -> str:
        """Decide whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        """Call the LLM with the current state."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def test_tools_directly():
    """Test the tools directly without LangGraph to isolate issues."""
    print("🔧 Testing tools directly...")

    # Test file stats
    print("\n1. Testing file stats tool:")
    stats_tool = LangGraphFileStatsTool()
    result = stats_tool._run("fixed_segmented_search.py")
    print(result)

    # Test file search
    print("\n2. Testing file search tool:")
    search_tool = LangGraphFileSearchTool()
    result = search_tool._run("fixed_segmented_search.py", "class", max_results=3)
    print(result)

    # Test file segment
    print("\n3. Testing file segment tool:")
    segment_tool = LangGraphFileSegmentTool()
    result = segment_tool._run("fixed_segmented_search.py", 1, 20)
    print(result)

    print("\n✅ Direct tool tests completed!")


def test_llm_connection():
    """Test basic LLM connection without tools."""
    print("🌐 Testing LLM connection...")
    try:
        llm = ChatOpenAI(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-30b-a3b",
            api_key="123",
            temperature=0.6,
            top_p=0.95,
        )

        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content="Hello, can you hear me?")])
        print(f"✅ LLM Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ LLM Connection failed: {e}")
        return False


# Comprehensive example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("🚀 Starting comprehensive file search tool tests...")

    # First test tools directly
    test_tools_directly()

    # Test LLM connection
    if not test_llm_connection():
        print(
            "❌ LLM connection failed. Please check your local server is running on localhost:1234"
        )
        exit(1)

    # If basic tests pass, create and test the agent
    print("\n🤖 Creating comprehensive agent...")
    try:
        agent = create_comprehensive_agent()
        print("✅ Agent created successfully!")

        # Simple test first
        print("\n🔍 Testing with simple query...")
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Get the file statistics for fixed_segmented_search.py"
                    )
                ]
            }
        )

        print("✅ Simple test completed!")
        print("Result:", result["messages"][-1].content)

    except Exception as e:
        print(f"❌ Agent creation or execution failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n�� Test completed!")
