"""Source grounding functionality for mapping extractions to text locations."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

from .schemas import SourceSpan


@dataclass
class GroundingMatch:
    """Represents a potential grounding match."""

    span: SourceSpan
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'semantic'


class SourceGrounder:
    """Maps extracted entities back to their source locations in text."""

    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold

    def ground_extraction(
        self, text: str, extracted_value: str, context_window: int = 50
    ) -> List[GroundingMatch]:
        """Find all possible source locations for an extracted value."""
        matches = []

        # Try exact matching first
        exact_matches = self._find_exact_matches(text, extracted_value)
        matches.extend(exact_matches)

        # If no exact matches, try fuzzy matching
        if not exact_matches:
            fuzzy_matches = self._find_fuzzy_matches(text, extracted_value)
            matches.extend(fuzzy_matches)

        # If still no matches, try semantic matching (partial word matches)
        if not matches:
            semantic_matches = self._find_semantic_matches(text, extracted_value)
            matches.extend(semantic_matches)

        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    def ground_entities(
        self, text: str, entities: Dict[str, any], chunk_offset: int = 0
    ) -> Dict[str, List[SourceSpan]]:
        """Ground all entities from an extraction result."""
        grounded = {}

        for field_name, value in entities.items():
            if value is None:
                continue

            # Handle different value types
            values_to_ground = self._extract_groundable_values(value)

            field_spans = []
            for val_str in values_to_ground:
                matches = self.ground_extraction(text, val_str)
                if matches:
                    # Take the best match and adjust for chunk offset
                    best_match = matches[0]
                    adjusted_span = SourceSpan(
                        start=best_match.span.start + chunk_offset,
                        end=best_match.span.end + chunk_offset,
                        text=best_match.span.text,
                    )
                    field_spans.append(adjusted_span)

            grounded[field_name] = field_spans

        return grounded

    def _find_exact_matches(self, text: str, value: str) -> List[GroundingMatch]:
        """Find exact substring matches."""
        matches = []
        start = 0

        while True:
            pos = text.find(value, start)
            if pos == -1:
                break

            span = SourceSpan(start=pos, end=pos + len(value), text=value)

            matches.append(
                GroundingMatch(span=span, confidence=1.0, match_type="exact")
            )

            start = pos + 1

        return matches

    def _find_fuzzy_matches(self, text: str, value: str) -> List[GroundingMatch]:
        """Find fuzzy matches using sequence matching."""
        matches = []
        value_words = value.lower().split()

        if not value_words:
            return matches

        # Use sliding window to find similar sequences
        window_size = len(value) + 20  # Allow some extra context
        step = window_size // 2

        for i in range(0, len(text) - window_size + 1, step):
            window_text = text[i : i + window_size]
            similarity = SequenceMatcher(
                None, value.lower(), window_text.lower()
            ).ratio()

            if similarity >= self.fuzzy_threshold:
                # Find the best substring within this window
                best_start, best_end, best_sim = self._find_best_substring_match(
                    window_text, value
                )

                if best_sim >= self.fuzzy_threshold:
                    actual_start = i + best_start
                    actual_end = i + best_end

                    span = SourceSpan(
                        start=actual_start,
                        end=actual_end,
                        text=text[actual_start:actual_end],
                    )

                    matches.append(
                        GroundingMatch(
                            span=span, confidence=best_sim, match_type="fuzzy"
                        )
                    )

        return matches

    def _find_semantic_matches(self, text: str, value: str) -> List[GroundingMatch]:
        """Find semantic matches by looking for key words."""
        matches = []
        value_words = set(
            word.lower().strip(".,!?;:") for word in value.split() if len(word) > 2
        )  # Skip short words

        if not value_words:
            return matches

        # Find regions with high word overlap
        text_words = text.split()
        window_size = max(5, len(value.split()) * 2)

        for i in range(len(text_words) - window_size + 1):
            window_words = set(
                word.lower().strip(".,!?;:") for word in text_words[i : i + window_size]
            )

            # Calculate word overlap
            overlap = len(value_words.intersection(window_words))
            similarity = overlap / len(value_words) if value_words else 0

            if similarity > 0.5:  # At least 50% word overlap
                # Find character positions
                start_pos = text.find(text_words[i])
                if start_pos != -1:
                    end_word = text_words[min(i + window_size - 1, len(text_words) - 1)]
                    end_pos = text.rfind(end_word) + len(end_word)

                    span = SourceSpan(
                        start=start_pos, end=end_pos, text=text[start_pos:end_pos]
                    )

                    matches.append(
                        GroundingMatch(
                            span=span,
                            confidence=similarity
                            * 0.8,  # Lower confidence for semantic matches
                            match_type="semantic",
                        )
                    )

        return matches

    def _find_best_substring_match(
        self, window_text: str, value: str
    ) -> Tuple[int, int, float]:
        """Find the best matching substring within a window."""
        best_sim = 0
        best_start = 0
        best_end = len(value)

        for start in range(len(window_text) - len(value) + 1):
            end = start + len(value)
            substring = window_text[start:end]
            similarity = SequenceMatcher(None, value.lower(), substring.lower()).ratio()

            if similarity > best_sim:
                best_sim = similarity
                best_start = start
                best_end = end

        return best_start, best_end, best_sim

    def _extract_groundable_values(self, value: any) -> List[str]:
        """Extract string values that can be grounded from various data types."""
        if isinstance(value, str):
            return [value] if value.strip() else []
        elif isinstance(value, (int, float, bool)):
            return [str(value)]
        elif isinstance(value, list):
            groundable = []
            for item in value:
                groundable.extend(self._extract_groundable_values(item))
            return groundable
        elif isinstance(value, dict):
            groundable = []
            for v in value.values():
                groundable.extend(self._extract_groundable_values(v))
            return groundable
        else:
            # Try to convert to string
            try:
                return [str(value)]
            except:
                return []

    def validate_grounding(
        self, text: str, span: SourceSpan, expected_value: str
    ) -> float:
        """Validate that a source span correctly grounds the expected value."""
        if span.start < 0 or span.end > len(text) or span.start >= span.end:
            return 0.0

        actual_text = text[span.start : span.end]
        if actual_text != span.text:
            return 0.0  # Span text doesn't match actual text

        # Calculate similarity between span text and expected value
        similarity = SequenceMatcher(
            None, expected_value.lower(), actual_text.lower()
        ).ratio()

        return similarity
