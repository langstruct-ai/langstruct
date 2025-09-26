"""Parallel processing utilities for LangStruct.

This module provides utilities for processing multiple documents or queries
in parallel, with rate limiting, error handling, and progress tracking.
"""

import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Ensures we don't exceed API rate limits by spacing out calls.
    """

    def __init__(self, calls_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
        self.last_call = 0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        if self.interval <= 0:
            return

        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()

    def __enter__(self):
        """Context manager entry."""
        self.wait_if_needed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False


@dataclass
class ProcessingResult:
    """Result from parallel processing, including successes and failures."""

    successful: List[Tuple[int, Any]] = field(default_factory=list)
    failed: List[Tuple[int, Exception]] = field(default_factory=list)

    @property
    def all_successful(self) -> bool:
        """Check if all items processed successfully."""
        return len(self.failed) == 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = len(self.successful) + len(self.failed)
        return (len(self.successful) / total * 100) if total > 0 else 100.0

    def get_results(self) -> List[Any]:
        """Get just the successful results in order."""
        # Sort by index to maintain order
        sorted_results = sorted(self.successful, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    def raise_if_failed(self):
        """Raise an exception if any items failed."""
        if self.failed:
            error_msgs = []
            for idx, exc in self.failed[:5]:  # Show first 5 errors
                error_msgs.append(f"  Item {idx}: {str(exc)}")

            msg = f"Failed to process {len(self.failed)} items:\n" + "\n".join(
                error_msgs
            )
            if len(self.failed) > 5:
                msg += f"\n  ... and {len(self.failed) - 5} more errors"

            raise RuntimeError(msg)


class ParallelProcessor:
    """Handles parallel processing with rate limiting and error handling."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        rate_limit: Optional[int] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize parallel processor.

        Args:
            max_workers: Maximum number of parallel workers
            rate_limit: API calls per minute (0 or None for no limit)
            retry_attempts: Number of retry attempts for failed items
            retry_delay: Delay between retries in seconds
        """
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def process_batch(
        self,
        items: List[T],
        process_fn: Callable[[T], Any],
        show_progress: bool = False,
        desc: Optional[str] = None,
    ) -> ProcessingResult:
        """Process items in parallel with error handling.

        Args:
            items: List of items to process
            process_fn: Function to process each item
            show_progress: Whether to show progress bar (requires tqdm)
            desc: Description for progress bar

        Returns:
            ProcessingResult with successful and failed items
        """
        if not items:
            return ProcessingResult()

        # Determine number of workers
        max_workers = self.max_workers or min(10, len(items))
        max_workers = min(
            max_workers, len(items)
        )  # Don't create more workers than items

        # Setup progress tracking
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(total=len(items), desc=desc or "Processing")
            except ImportError:
                warnings.warn(
                    "tqdm not installed. Install with: pip install langstruct[parallel]",
                    UserWarning,
                )

        result = ProcessingResult()

        # Process items in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx: Dict[Future, int] = {}

            for idx, item in enumerate(items):
                future = executor.submit(self._process_with_retry, item, process_fn)
                future_to_idx[future] = idx

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]

                try:
                    # Get result with timeout
                    output = future.result(timeout=60)
                    result.successful.append((idx, output))
                except Exception as e:
                    result.failed.append((idx, e))

                # Update progress
                if progress_bar:
                    progress_bar.update(1)

        # Close progress bar
        if progress_bar:
            progress_bar.close()

        return result

    def _process_with_retry(self, item: T, process_fn: Callable[[T], Any]) -> Any:
        """Process single item with retry logic and rate limiting.

        Args:
            item: Item to process
            process_fn: Function to process the item

        Returns:
            Result from process_fn

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                # Apply rate limiting
                if self.rate_limiter:
                    with self.rate_limiter:
                        return process_fn(item)
                else:
                    return process_fn(item)

            except Exception as e:
                last_exception = e

                # Don't retry on certain errors
                error_msg = str(e).lower()
                if any(
                    msg in error_msg
                    for msg in ["invalid api key", "authentication", "unauthorized"]
                ):
                    raise

                # Exponential backoff for retries
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(
                f"Failed to process item after {self.retry_attempts} attempts"
            )


def with_parallel_support(func: Callable) -> Callable:
    """Decorator to add parallel processing support to a method.

    This decorator allows a method to handle both single items and lists,
    automatically using parallel processing for lists.
    """

    @wraps(func)
    def wrapper(self, items, *args, **kwargs):
        # Extract parallel-specific kwargs
        max_workers = kwargs.pop("max_workers", None)
        show_progress = kwargs.pop("show_progress", False)
        rate_limit = kwargs.pop("rate_limit", None)
        retry_failed = kwargs.pop("retry_failed", True)

        # Handle single vs batch
        if isinstance(items, list):
            # Use parallel processing for lists
            processor = ParallelProcessor(
                max_workers=max_workers, rate_limit=rate_limit
            )

            # Create process function that includes the remaining kwargs
            def process_fn(item):
                return func(self, item, *args, **kwargs)

            # Process in parallel
            result = processor.process_batch(
                items=items,
                process_fn=process_fn,
                show_progress=show_progress,
                desc=f"{func.__name__}",
            )

            # Handle failures
            if retry_failed:
                result.raise_if_failed()
            elif result.failed:
                warnings.warn(
                    f"{len(result.failed)} items failed during {func.__name__}. "
                    f"Success rate: {result.success_rate:.1f}%",
                    UserWarning,
                )

            return result.get_results()
        else:
            # Single item - process directly
            return func(self, items, *args, **kwargs)

    return wrapper
