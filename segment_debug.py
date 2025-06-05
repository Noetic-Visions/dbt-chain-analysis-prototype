#!/usr/bin/env python3
"""Debug test specifically for get_file_segment method."""

import os
import tempfile

# Create test content
test_content = """
# Test Python File
import os
import sys

def function_one():
    print("Hello World")
    return True

class TestClass:
    def __init__(self):
        self.config = "test"

    def method_one(self):
        return "test result"

def function_two(param):
    if param > 10:
        return "large"
    return "small"

# End of file
"""

print("=== File Segment Debug Test ===")

# Create temporary file
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(test_content)
    test_file = f.name
    print(f"Test file created: {test_file}")

try:
    from segmented_file_search import AgentFileSearchTool

    print("Import successful")

    agent_tool = AgentFileSearchTool()
    print("Agent tool created")

    print("\nTesting get_file_segment method...")
    print("Calling get_file_segment(test_file, 1, 10)...")

    segment = agent_tool.get_file_segment(test_file, 1, 10)
    print("get_file_segment completed!")
    print(f"Success: {segment['success']}")
    print(f"Content length: {len(segment['content'])} chars")
    print(f"First 50 chars: {segment['content'][:50]}...")

    print("\nTesting edge case - larger segment...")
    segment2 = agent_tool.get_file_segment(test_file, 5, 20)
    print(f"Second segment success: {segment2['success']}")

    print("\nTesting invalid range...")
    segment3 = agent_tool.get_file_segment(test_file, 100, 200)
    print(f"Invalid range success: {segment3['success']}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

finally:
    os.unlink(test_file)
    print("Segment debug test completed")
