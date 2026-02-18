"""Shared constants for langstruct core modules."""

# Maximum number of retries when LLM returns invalid JSON.
# Each retry feeds the parse error back to the LLM as context.
MAX_PARSE_RETRIES = 3
