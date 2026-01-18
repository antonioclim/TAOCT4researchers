#!/usr/bin/env python3
"""Solution for Exercise 09: Checkpoint Recovery System.

This module provides reference implementations for checkpoint-based
recovery in long-running computations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class CheckpointData:
    """Data stored in a checkpoint."""
    
    state: dict[str, Any] = field(default_factory=dict)
    completed_items: set[str] = field(default_factory=set)
    failed_items: dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages checkpoint persistence for resumable computations."""
    
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        backup_count: int = 3,
        use_pickle: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._backup_count = backup_count
        self._use_pickle = use_pickle
        self._data = CheckpointData()
    
    @property
    def data(self) -> CheckpointData:
        return self._data
    
    def save(self) -> None:
        """Save checkpoint to file atomically."""
        self._data.last_update = time.time()
        self._rotate_backups()
        
        # Prepare serialisable data
        save_data = {
            "state": self._data.state,
            "completed_items": list(self._data.completed_items),
            "failed_items": self._data.failed_items,
            "start_time": self._data.start_time,
            "last_update": self._data.last_update,
            "metadata": self._data.metadata,
        }
        
        # Write to temp file then rename (atomic)
        temp_path = self._checkpoint_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(save_data, indent=2))
        temp_path.rename(self._checkpoint_path)
        
        logger.info("Checkpoint saved to %s", self._checkpoint_path)
    
    def load(self) -> bool:
        """Load checkpoint from file."""
        if not self._checkpoint_path.exists():
            return False
        
        try:
            content = self._checkpoint_path.read_text()
            save_data = json.loads(content)
            
            self._data = CheckpointData(
                state=save_data.get("state", {}),
                completed_items=set(save_data.get("completed_items", [])),
                failed_items=save_data.get("failed_items", {}),
                start_time=save_data.get("start_time", time.time()),
                last_update=save_data.get("last_update", time.time()),
                metadata=save_data.get("metadata", {}),
            )
            
            logger.info("Checkpoint loaded from %s", self._checkpoint_path)
            return True
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return False
    
    def _rotate_backups(self) -> None:
        """Rotate backup files."""
        if not self._checkpoint_path.exists():
            return
        
        # Shift existing backups
        for i in range(self._backup_count, 0, -1):
            old_path = self._checkpoint_path.with_suffix(f".json.{i}")
            new_path = self._checkpoint_path.with_suffix(f".json.{i + 1}")
            if old_path.exists():
                if i == self._backup_count:
                    old_path.unlink()
                else:
                    old_path.rename(new_path)
        
        # Move current to .1
        backup_path = self._checkpoint_path.with_suffix(f".json.1")
        if self._checkpoint_path.exists():
            self._checkpoint_path.rename(backup_path)
    
    def mark_completed(self, item_id: str) -> None:
        self._data.completed_items.add(item_id)
    
    def mark_failed(self, item_id: str, error: str) -> None:
        self._data.failed_items[item_id] = error
    
    def is_completed(self, item_id: str) -> bool:
        return item_id in self._data.completed_items
    
    def is_failed(self, item_id: str) -> bool:
        return item_id in self._data.failed_items
    
    def get_progress(self) -> tuple[int, int, int]:
        completed = len(self._data.completed_items)
        failed = len(self._data.failed_items)
        total = completed + failed
        return completed, failed, total
    
    def clear(self) -> None:
        self._data = CheckpointData()
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    
    total_items: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.completed + self.failed
        if total == 0:
            return 100.0
        return (self.completed / total) * 100


class ResumableBatchProcessor(Generic[T, R]):
    """Processes items in batches with checkpoint-based recovery."""
    
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
        self._items = items
        self._process_func = process_func
        self._item_id_func = item_id_func
        self._checkpoint_interval = checkpoint_interval
        self._continue_on_error = continue_on_error
        self._max_failures = max_failures
        
        self._checkpoint = CheckpointManager(checkpoint_path)
        self._stats = ProcessingStats(total_items=len(items))
        self._results: dict[str, R] = {}
    
    def run(self) -> dict[str, R]:
        """Run batch processing."""
        self._checkpoint.load()
        start_time = time.time()
        
        items_since_checkpoint = 0
        
        for item in self._items:
            item_id = self._item_id_func(item)
            
            if self._checkpoint.is_completed(item_id):
                self._stats.skipped += 1
                continue
            
            try:
                result = self._process_func(item)
                self._results[item_id] = result
                self._checkpoint.mark_completed(item_id)
                self._stats.completed += 1
                
            except Exception as e:
                self._checkpoint.mark_failed(item_id, str(e))
                self._stats.failed += 1
                
                if self._max_failures and self._stats.failed >= self._max_failures:
                    logger.error("Max failures reached, stopping")
                    break
                
                if not self._continue_on_error:
                    raise
            
            items_since_checkpoint += 1
            if items_since_checkpoint >= self._checkpoint_interval:
                self._checkpoint.save()
                items_since_checkpoint = 0
        
        self._checkpoint.save()
        self._stats.elapsed_time = time.time() - start_time
        
        return self._results
    
    def get_stats(self) -> ProcessingStats:
        return self._stats
    
    def get_failed_items(self) -> dict[str, str]:
        return dict(self._checkpoint.data.failed_items)
    
    def retry_failed(self) -> dict[str, R]:
        """Retry processing of previously failed items."""
        failed_ids = set(self._checkpoint.data.failed_items.keys())
        results: dict[str, R] = {}
        
        for item in self._items:
            item_id = self._item_id_func(item)
            
            if item_id not in failed_ids:
                continue
            
            try:
                result = self._process_func(item)
                results[item_id] = result
                self._checkpoint.mark_completed(item_id)
                del self._checkpoint.data.failed_items[item_id]
                
            except Exception as e:
                self._checkpoint.data.failed_items[item_id] = str(e)
        
        self._checkpoint.save()
        return results


def chunked_iterator(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Iterate over items in chunks."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def process_with_checkpoints(
    items: list[T],
    process_func: Callable[[T], R],
    checkpoint_path: Path,
    *,
    chunk_size: int = 100,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[R], ProcessingStats]:
    """Process items with chunked checkpointing."""
    manager = CheckpointManager(checkpoint_path)
    manager.load()
    
    stats = ProcessingStats(total_items=len(items))
    results: list[R] = []
    
    start_time = time.time()
    completed_chunks = manager.data.state.get("completed_chunks", 0)
    
    chunks = list(chunked_iterator(items, chunk_size))
    
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx < completed_chunks:
            # Load results from checkpoint
            chunk_results = manager.data.state.get(f"chunk_{chunk_idx}_results", [])
            results.extend(chunk_results)
            stats.skipped += len(chunk)
            continue
        
        chunk_results: list[R] = []
        for item in chunk:
            try:
                result = process_func(item)
                chunk_results.append(result)
                stats.completed += 1
            except Exception as e:
                logger.error("Failed to process item: %s", e)
                stats.failed += 1
        
        results.extend(chunk_results)
        
        # Save checkpoint
        manager.data.state["completed_chunks"] = chunk_idx + 1
        manager.data.state[f"chunk_{chunk_idx}_results"] = chunk_results
        manager.save()
        
        if on_progress:
            total_processed = stats.completed + stats.failed + stats.skipped
            on_progress(total_processed, len(items))
    
    stats.elapsed_time = time.time() - start_time
    return results, stats


def main() -> None:
    """Verify solution implementations."""
    import tempfile
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing CheckpointManager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"
        
        manager = CheckpointManager(path)
        manager.mark_completed("item1")
        manager.mark_completed("item2")
        manager.mark_failed("item3", "test error")
        manager.data.state["key"] = "value"
        manager.save()
        
        manager2 = CheckpointManager(path)
        assert manager2.load()
        assert manager2.is_completed("item1")
        assert manager2.is_completed("item2")
        assert manager2.is_failed("item3")
        assert manager2.data.state["key"] == "value"
        
        completed, failed, total = manager2.get_progress()
        assert completed == 2
        assert failed == 1
        
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
        
        progress_calls: list[tuple[int, int]] = []
        
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
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
