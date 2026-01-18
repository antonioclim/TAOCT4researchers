#!/usr/bin/env python3
"""Exercise 09: Checkpoint Recovery System (Hard).

This exercise implements a checkpoint-based recovery system for
long-running computations.

Learning Objectives:
    - Implement checkpoint save/restore mechanisms
    - Handle partial failures in batch processing
    - Design resumable computation patterns

Estimated Time: 20 minutes
Difficulty: Hard (★★★)

Instructions:
    1. Implement CheckpointManager for state persistence
    2. Build resumable batch processor
    3. Add progress tracking and recovery
    4. Run tests to verify your implementation
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# TASK 1: Checkpoint Manager
# =============================================================================

@dataclass
class CheckpointData:
    """Data stored in a checkpoint.
    
    Attributes:
        state: Arbitrary state dictionary.
        completed_items: Set of completed item identifiers.
        failed_items: Dict of item ID to error message.
        start_time: When processing started.
        last_update: When checkpoint was last updated.
        metadata: Additional metadata.
    """
    
    state: dict[str, Any] = field(default_factory=dict)
    completed_items: set[str] = field(default_factory=set)
    failed_items: dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages checkpoint persistence for resumable computations.
    
    Provides atomic save/restore operations with automatic backup.
    
    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mktemp(suffix=".json"))
        >>> manager = CheckpointManager(path)
        >>> manager.mark_completed("item1")
        >>> manager.save()
        >>> manager2 = CheckpointManager(path)
        >>> manager2.load()
        True
        >>> manager2.is_completed("item1")
        True
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        backup_count: int = 3,
        use_pickle: bool = False,
    ) -> None:
        """Initialise checkpoint manager.
        
        Args:
            checkpoint_path: Path for checkpoint file.
            backup_count: Number of backup files to keep.
            use_pickle: Use pickle instead of JSON (for non-serialisable data).
        """
        # TODO: Store parameters
        # TODO: Initialise CheckpointData
        pass
    
    @property
    def data(self) -> CheckpointData:
        """Current checkpoint data."""
        # TODO: Return checkpoint data
        pass
    
    def save(self) -> None:
        """Save checkpoint to file atomically.
        
        Uses write-to-temp-then-rename pattern for atomicity.
        Rotates backup files before saving.
        """
        # TODO: Implement atomic save
        # 1. Update last_update timestamp
        # 2. Rotate existing backups
        # 3. Write to temp file
        # 4. Rename temp file to checkpoint path
        pass
    
    def load(self) -> bool:
        """Load checkpoint from file.
        
        Returns:
            True if checkpoint was loaded, False if not found.
        """
        # TODO: Implement load
        # 1. Check if file exists
        # 2. Load and parse file
        # 3. Restore CheckpointData
        # 4. Return success/failure
        pass
    
    def _rotate_backups(self) -> None:
        """Rotate backup files.
        
        Keeps backup_count previous versions.
        """
        # TODO: Implement backup rotation
        # checkpoint.json -> checkpoint.json.1 -> checkpoint.json.2 -> ...
        pass
    
    def mark_completed(self, item_id: str) -> None:
        """Mark an item as completed.
        
        Args:
            item_id: Unique identifier for the item.
        """
        # TODO: Add to completed_items set
        pass
    
    def mark_failed(self, item_id: str, error: str) -> None:
        """Mark an item as failed.
        
        Args:
            item_id: Unique identifier for the item.
            error: Error message or description.
        """
        # TODO: Add to failed_items dict
        pass
    
    def is_completed(self, item_id: str) -> bool:
        """Check if item is completed.
        
        Args:
            item_id: Item identifier to check.
            
        Returns:
            True if item was completed.
        """
        # TODO: Check completed_items
        pass
    
    def is_failed(self, item_id: str) -> bool:
        """Check if item previously failed.
        
        Args:
            item_id: Item identifier to check.
            
        Returns:
            True if item previously failed.
        """
        # TODO: Check failed_items
        pass
    
    def get_progress(self) -> tuple[int, int, int]:
        """Get progress statistics.
        
        Returns:
            Tuple of (completed_count, failed_count, total_attempted).
        """
        # TODO: Return counts
        pass
    
    def clear(self) -> None:
        """Clear checkpoint data and delete file."""
        # TODO: Reset CheckpointData and delete file
        pass


# =============================================================================
# TASK 2: Resumable Batch Processor
# =============================================================================

@dataclass
class ProcessingStats:
    """Statistics for batch processing.
    
    Attributes:
        total_items: Total items to process.
        completed: Successfully processed items.
        failed: Items that failed processing.
        skipped: Items skipped (already completed).
        elapsed_time: Total processing time.
    """
    
    total_items: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.completed + self.failed
        if total == 0:
            return 100.0
        return (self.completed / total) * 100


class ResumableBatchProcessor(Generic[T, R]):
    """Processes items in batches with checkpoint-based recovery.
    
    Supports resuming from failures and tracking progress.
    
    Example:
        >>> def process(x):
        ...     return x * 2
        >>> processor = ResumableBatchProcessor(
        ...     items=[1, 2, 3],
        ...     process_func=process,
        ...     checkpoint_path=Path("checkpoint.json"),
        ... )
        >>> results = processor.run()
    """
    
    def __init__(
        self,
        items: list[T],
        process_func: Callable[[T], R],
        checkpoint_path: Path,
        *,
        item_id_func: Callable[[T], str] = str,
        checkpoint_interval: int = 10,
        continue_on_error: bool = True,
        max_failures: int | None = None,
    ) -> None:
        """Initialise batch processor.
        
        Args:
            items: Items to process.
            process_func: Function to apply to each item.
            checkpoint_path: Path for checkpoint file.
            item_id_func: Function to get unique ID from item.
            checkpoint_interval: Items between checkpoints.
            continue_on_error: Continue after item failures.
            max_failures: Stop after this many failures (None = unlimited).
        """
        # TODO: Store all parameters
        # TODO: Create CheckpointManager
        pass
    
    def run(self) -> dict[str, R]:
        """Run batch processing.
        
        Returns:
            Dictionary mapping item IDs to results.
        """
        # TODO: Implement processing
        # 1. Load existing checkpoint
        # 2. Track stats with ProcessingStats
        # 3. For each item:
        #    - Skip if already completed
        #    - Process and handle exceptions
        #    - Save checkpoint at intervals
        # 4. Save final checkpoint
        # 5. Return results
        pass
    
    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        # TODO: Return stats
        pass
    
    def get_failed_items(self) -> dict[str, str]:
        """Get items that failed processing.
        
        Returns:
            Dictionary mapping item IDs to error messages.
        """
        # TODO: Return failed items from checkpoint
        pass
    
    def retry_failed(self) -> dict[str, R]:
        """Retry processing of previously failed items.
        
        Returns:
            Dictionary mapping item IDs to results for retried items.
        """
        # TODO: Process only items in failed_items
        # TODO: Remove from failed_items on success
        pass


# =============================================================================
# TASK 3: Chunked Processing with Recovery
# =============================================================================

def chunked_iterator(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Iterate over items in chunks.
    
    Args:
        items: Items to chunk.
        chunk_size: Size of each chunk.
        
    Yields:
        Lists of items, each up to chunk_size.
    """
    # TODO: Implement chunking
    pass


def process_with_checkpoints(
    items: list[T],
    process_func: Callable[[T], R],
    checkpoint_path: Path,
    *,
    chunk_size: int = 100,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[R], ProcessingStats]:
    """Process items with chunked checkpointing.
    
    Processes items in chunks, saving a checkpoint after each chunk.
    Can resume from the last completed chunk on restart.
    
    Args:
        items: Items to process.
        process_func: Processing function.
        checkpoint_path: Path for checkpoint file.
        chunk_size: Items per checkpoint.
        on_progress: Optional callback(completed, total).
        
    Returns:
        Tuple of (results_list, processing_stats).
        
    Example:
        >>> def double(x):
        ...     return x * 2
        >>> results, stats = process_with_checkpoints(
        ...     [1, 2, 3, 4, 5],
        ...     double,
        ...     Path("checkpoint.json"),
        ...     chunk_size=2,
        ... )
        >>> results
        [2, 4, 6, 8, 10]
    """
    # TODO: Implement chunked processing
    # 1. Create checkpoint manager
    # 2. Load existing checkpoint
    # 3. Determine which chunks are completed
    # 4. Process remaining chunks
    # 5. Save checkpoint after each chunk
    # 6. Report progress via callback
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    import tempfile
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing CheckpointManager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"
        
        # Test save and load
        manager = CheckpointManager(path)
        manager.mark_completed("item1")
        manager.mark_completed("item2")
        manager.mark_failed("item3", "test error")
        manager.data.state["key"] = "value"
        manager.save()
        
        # Load in new manager
        manager2 = CheckpointManager(path)
        assert manager2.load()
        assert manager2.is_completed("item1")
        assert manager2.is_completed("item2")
        assert manager2.is_failed("item3")
        assert manager2.data.state["key"] == "value"
        
        # Test progress
        completed, failed, total = manager2.get_progress()
        assert completed == 2
        assert failed == 1
        
        # Test clear
        manager2.clear()
        assert not path.exists()
    print("  ✓ CheckpointManager passed")
    
    print("Testing ResumableBatchProcessor...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "batch_checkpoint.json"
        
        items = list(range(10))
        
        def process(x: int) -> int:
            if x == 5:
                raise ValueError(f"Failed on {x}")
            return x * 2
        
        processor = ResumableBatchProcessor(
            items=items,
            process_func=process,
            checkpoint_path=path,
            continue_on_error=True,
        )
        
        results = processor.run()
        stats = processor.get_stats()
        
        assert stats.completed == 9
        assert stats.failed == 1
        assert len(results) == 9
        assert "5" in processor.get_failed_items()
    print("  ✓ ResumableBatchProcessor passed")
    
    print("Testing chunked_iterator...")
    items = [1, 2, 3, 4, 5]
    chunks = list(chunked_iterator(items, 2))
    assert chunks == [[1, 2], [3, 4], [5]]
    print("  ✓ chunked_iterator passed")
    
    print("Testing process_with_checkpoints...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chunk_checkpoint.json"
        
        progress_calls = []
        
        def track_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))
        
        results, stats = process_with_checkpoints(
            items=[1, 2, 3, 4, 5],
            process_func=lambda x: x * 2,
            checkpoint_path=path,
            chunk_size=2,
            on_progress=track_progress,
        )
        
        assert results == [2, 4, 6, 8, 10]
        assert stats.completed == 5
        assert len(progress_calls) > 0
    print("  ✓ process_with_checkpoints passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
