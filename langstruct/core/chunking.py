"""Smart text chunking for long document processing."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tiktoken
from pydantic import BaseModel


@dataclass(frozen=True)
class TextChunk:
    """Represents a chunk of text with position information."""

    id: int
    text: str
    start_offset: int
    end_offset: int
    overlap_start: int = 0  # Characters of overlap with previous chunk
    overlap_end: int = 0  # Characters of overlap with next chunk


class ChunkingConfig(BaseModel):
    """Configuration for text chunking strategy."""

    max_tokens: int = 2000
    overlap_tokens: int = 200
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    min_chunk_tokens: int = 100
    encoding_name: str = "cl100k_base"  # GPT-4 encoding


class TextChunker:
    """Intelligent text chunker that maintains context and source positions."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.encoding = tiktoken.get_encoding(self.config.encoding_name)

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks with preserved boundaries."""
        if not text.strip():
            return []

        # First check if the entire text fits in one chunk
        if self._count_tokens(text) <= self.config.max_tokens:
            return [TextChunk(id=0, text=text, start_offset=0, end_offset=len(text))]

        # Find natural break points
        break_points = self._find_break_points(text)

        # Create chunks respecting break points and token limits
        chunks = []
        current_start = 0
        chunk_id = 0

        while current_start < len(text):
            chunk_end = self._find_chunk_end(text, current_start, break_points)

            # Extract chunk text with overlap
            overlap_start = self._calculate_overlap_start(current_start, chunks)
            chunk_text = text[overlap_start:chunk_end]

            # Skip if chunk is too small (unless it's the last chunk)
            if self._count_tokens(
                chunk_text
            ) < self.config.min_chunk_tokens and chunk_end < len(text):
                current_start = chunk_end
                continue

            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    start_offset=current_start,
                    end_offset=chunk_end,
                    overlap_start=current_start - overlap_start,
                    overlap_end=0,  # Will be calculated for next chunk
                )
            )

            # Update overlap_end for previous chunk
            if len(chunks) > 1:
                prev_chunk = chunks[-2]
                overlap_chars = max(0, prev_chunk.end_offset - current_start)
                chunks[-2] = prev_chunk.__class__(
                    **{**prev_chunk.__dict__, "overlap_end": overlap_chars}
                )

            current_start = chunk_end
            chunk_id += 1

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured encoding."""
        return len(self.encoding.encode(text))

    def _find_break_points(self, text: str) -> List[int]:
        """Find natural break points in text (paragraphs, sentences)."""
        break_points = [0]

        if self.config.preserve_paragraphs:
            # Find paragraph breaks (double newlines)
            for match in re.finditer(r"\n\s*\n", text):
                break_points.append(match.end())

        if self.config.preserve_sentences:
            # Find sentence breaks
            sentence_pattern = r"[.!?]+\s+"
            for match in re.finditer(sentence_pattern, text):
                break_points.append(match.end())

        break_points.append(len(text))
        return sorted(set(break_points))

    def _find_chunk_end(self, text: str, start: int, break_points: List[int]) -> int:
        """Find the best end position for a chunk starting at start."""
        max_chars = self._estimate_chars_for_tokens(self.config.max_tokens)
        ideal_end = start + max_chars

        if ideal_end >= len(text):
            return len(text)

        # Find the best break point before the ideal end
        best_break = ideal_end
        for break_point in break_points:
            if start < break_point <= ideal_end:
                best_break = break_point
            elif break_point > ideal_end:
                break

        # Ensure we don't exceed token limit
        chunk_text = text[start:best_break]
        while (
            self._count_tokens(chunk_text) > self.config.max_tokens
            and best_break > start + 1
        ):
            # Find previous break point
            prev_breaks = [bp for bp in break_points if start < bp < best_break]
            if prev_breaks:
                best_break = prev_breaks[-1]
            else:
                # Force break at character level
                best_break = start + max_chars // 2
                break
            chunk_text = text[start:best_break]

        return best_break

    def _calculate_overlap_start(
        self, current_start: int, chunks: List[TextChunk]
    ) -> int:
        """Calculate where to start the chunk including overlap."""
        if not chunks:
            return current_start

        overlap_chars = self._estimate_chars_for_tokens(self.config.overlap_tokens)
        overlap_start = max(0, current_start - overlap_chars)

        # Don't overlap beyond the start of the previous chunk
        prev_chunk = chunks[-1]
        overlap_start = max(overlap_start, prev_chunk.start_offset)

        return overlap_start

    def _estimate_chars_for_tokens(self, tokens: int) -> int:
        """Rough estimate of characters needed for token count."""
        # Average ~4 characters per token for English text
        return tokens * 4

    def merge_chunk_results(self, chunks: List[TextChunk], results: List[any]) -> any:
        """Merge results from multiple chunks, handling overlaps."""
        # This will be implemented when we have the extraction results structure
        # For now, return a placeholder
        if not results:
            return None
        return results[0]
