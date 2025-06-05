import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FileSegment:
    """Represents a segment of a file with line numbers and content."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    total_lines: Optional[int] = None


@dataclass
class SearchMatch:
    """Represents a search match within a file segment."""

    file_path: str
    line_number: int
    line_content: str
    segment: FileSegment


class FixedSegmentedFileSearchTool:
    """
    A file search tool that scans files in segments without subprocess calls.
    Uses only Python file operations for reliability.
    """

    def __init__(self, segment_size: int = 100, overlap: int = 10):
        """
        Initialize the segmented file search tool.

        Args:
            segment_size: Number of lines per segment
            overlap: Number of lines to overlap between segments (for context)
        """
        self.segment_size = segment_size
        self.overlap = overlap

    def get_file_line_count(self, file_path: str) -> int:
        """Get total number of lines in a file using Python."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def extract_segment(
        self, file_path: str, start_line: int, end_line: int
    ) -> FileSegment:
        """
        Extract a segment of lines from a file using Python.

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)

        Returns:
            FileSegment object containing the extracted content
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Convert to 0-indexed for slicing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)

            segment_lines = lines[start_idx:end_idx]
            content = "".join(segment_lines)

            return FileSegment(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=content,
                total_lines=len(lines),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract segment from {file_path}: {e}")

    def search_in_segment(
        self, segment: FileSegment, pattern: str, use_regex: bool = True
    ) -> List[SearchMatch]:
        """
        Search for a pattern within a file segment.

        Args:
            segment: FileSegment to search in
            pattern: Search pattern
            use_regex: Whether to treat pattern as regex

        Returns:
            List of SearchMatch objects
        """
        matches = []
        lines = segment.content.split("\n")

        for i, line in enumerate(lines):
            line_number = segment.start_line + i

            found = False
            if use_regex:
                try:
                    found = bool(re.search(pattern, line))
                except re.error:
                    # If regex is invalid, fall back to literal search
                    found = pattern in line
            else:
                found = pattern in line

            if found:
                matches.append(
                    SearchMatch(
                        file_path=segment.file_path,
                        line_number=line_number,
                        line_content=line,
                        segment=segment,
                    )
                )

        return matches

    def smart_segment_search(
        self, file_path: str, pattern: str, max_matches: int = 10
    ) -> List[SearchMatch]:
        """
        Smart segmented search that stops early if enough matches are found.

        Args:
            file_path: Path to the file to search
            pattern: Search pattern
            max_matches: Maximum number of matches to find before stopping

        Returns:
            List of SearchMatch objects (up to max_matches)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        total_lines = self.get_file_line_count(file_path)
        all_matches = []

        current_line = 1
        while current_line <= total_lines and len(all_matches) < max_matches:
            end_line = min(current_line + self.segment_size - 1, total_lines)

            # Extract segment and search
            segment = self.extract_segment(file_path, current_line, end_line)
            matches = self.search_in_segment(segment, pattern, use_regex=False)

            all_matches.extend(matches)

            # If we have enough matches, break early
            if len(all_matches) >= max_matches:
                all_matches = all_matches[:max_matches]
                break

            # Calculate next starting line with overlap
            next_line = end_line - self.overlap + 1

            # Prevent infinite loop: ensure we always advance
            if next_line <= current_line:
                next_line = current_line + 1

            current_line = next_line

        return all_matches

    def get_context_around_match(
        self, match: SearchMatch, context_lines: int = 3
    ) -> FileSegment:
        """
        Get additional context lines around a search match.

        Args:
            match: SearchMatch object
            context_lines: Number of lines before and after to include

        Returns:
            FileSegment with extended context
        """
        start_line = max(1, match.line_number - context_lines)
        end_line = match.line_number + context_lines

        return self.extract_segment(match.file_path, start_line, end_line)


class FixedAgentFileSearchTool:
    """
    Wrapper class that provides a clean interface for LLM agents.
    """

    def __init__(self):
        self.searcher = FixedSegmentedFileSearchTool(segment_size=100, overlap=10)

    def search_file(self, file_path: str, query: str, max_results: int = 5) -> Dict:
        """
        Main interface for LLM agents to search files.

        Args:
            file_path: Path to the file to search
            query: Search query/pattern
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:
            matches = self.searcher.smart_segment_search(
                file_path, query, max_matches=max_results
            )

            return {
                "success": True,
                "file_path": file_path,
                "query": query,
                "total_matches": len(matches),
                "matches": [
                    {
                        "line_number": match.line_number,
                        "content": match.line_content.strip(),
                        "file_path": match.file_path,
                    }
                    for match in matches
                ],
                "message": f"Found {len(matches)} matches in {file_path}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "query": query,
            }

    def get_file_segment(self, file_path: str, start_line: int, end_line: int) -> Dict:
        """
        Get a specific segment of a file for detailed analysis.

        Args:
            file_path: Path to the file
            start_line: Starting line number
            end_line: Ending line number

        Returns:
            Dictionary with segment content and metadata
        """
        try:
            segment = self.searcher.extract_segment(file_path, start_line, end_line)

            return {
                "success": True,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "content": segment.content,
                "total_file_lines": segment.total_lines,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}
