#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Amortised Analysis (Hard)
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Amortised analysis provides a more accurate picture of algorithm performance
when occasional expensive operations are balanced by many cheap operations.
This is crucial for understanding data structures like dynamic arrays, hash
tables and splay trees. This exercise develops advanced analytical skills
for realistic performance evaluation.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Apply the aggregate method for amortised analysis
2. Implement and analyse the accounting method
3. Use the potential method for complex data structures
4. Verify theoretical amortised bounds empirically

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 60 minutes
- Total: 80 minutes

DIFFICULTY: ⭐⭐⭐⭐⭐ (Hard)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DYNAMIC ARRAY WITH AMORTISED O(1) APPEND
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicArray(Generic[T]):
    """
    A dynamic array implementation demonstrating amortised analysis.
    
    The key insight: Although resize operations are O(n), they happen
    infrequently enough that the amortised cost per append is O(1).
    
    Attributes:
        _data: Internal fixed-size array.
        _size: Number of elements currently stored.
        _capacity: Current capacity of internal array.
        _resize_count: Number of resize operations performed.
        _total_copies: Total elements copied during all resizes.
    """
    
    _data: list[T | None] = field(default_factory=lambda: [None] * 4)
    _size: int = 0
    _capacity: int = 4
    _resize_count: int = 0
    _total_copies: int = 0
    _growth_factor: float = 2.0
    
    def append(self, item: T) -> int:
        """
        Append item to the array.
        
        Args:
            item: Element to append.
            
        Returns:
            Number of element copies made (0 if no resize, n if resize).
            
        Amortised Analysis (Aggregate Method):
            Over n appends starting from empty:
            - Most appends: O(1) - just place element
            - Resizes occur at: 4, 8, 16, 32, ... elements
            - Copies: 4 + 8 + 16 + ... + n ≤ 2n
            - Total work: n (placements) + 2n (copies) = 3n
            - Amortised: 3n / n = O(1) per append
        """
        copies_made = 0
        
        if self._size >= self._capacity:
            copies_made = self._resize()
        
        self._data[self._size] = item
        self._size += 1
        
        return copies_made
    
    def _resize(self) -> int:
        """
        Double the capacity of the internal array.
        
        Returns:
            Number of elements copied.
        """
        new_capacity = int(self._capacity * self._growth_factor)
        new_data: list[T | None] = [None] * new_capacity
        
        # Copy existing elements
        for i in range(self._size):
            new_data[i] = self._data[i]
        
        self._data = new_data
        self._capacity = new_capacity
        self._resize_count += 1
        self._total_copies += self._size
        
        return self._size
    
    def get(self, index: int) -> T:
        """Get element at index."""
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size})")
        return self._data[index]  # type: ignore
    
    def __len__(self) -> int:
        """Return current size."""
        return self._size
    
    def stats(self) -> dict[str, Any]:
        """Return statistics about resize operations."""
        return {
            "size": self._size,
            "capacity": self._capacity,
            "resize_count": self._resize_count,
            "total_copies": self._total_copies,
            "amortised_copies_per_append": (
                self._total_copies / self._size if self._size > 0 else 0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MULTIPOP STACK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultipopStack(Generic[T]):
    """
    A stack with a multipop operation for amortised analysis practice.
    
    Operations:
        - push(x): O(1) - add element to top
        - pop(): O(1) - remove top element
        - multipop(k): O(min(k, n)) - pop k elements
        
    Although multipop is O(n) worst case, the amortised cost of any
    sequence of operations is O(1) per operation.
    
    Accounting Method Analysis:
        - Assign cost $2 to each push
        - $1 pays for the push itself
        - $1 is "credit" stored with the element
        - When popped (singly or via multipop), use the stored $1
        - Every operation costs at most $2 amortised
    """
    
    _items: list[T] = field(default_factory=list)
    _push_count: int = 0
    _pop_count: int = 0
    _multipop_count: int = 0
    
    def push(self, item: T) -> None:
        """Push item onto the stack."""
        self._items.append(item)
        self._push_count += 1
    
    def pop(self) -> T:
        """Pop and return top item."""
        if not self._items:
            raise IndexError("Pop from empty stack")
        self._pop_count += 1
        return self._items.pop()
    
    def multipop(self, k: int) -> list[T]:
        """
        Pop and return up to k items.
        
        Args:
            k: Maximum number of items to pop.
            
        Returns:
            List of popped items (may be fewer than k if stack has fewer).
        """
        self._multipop_count += 1
        result: list[T] = []
        
        actual_pops = min(k, len(self._items))
        for _ in range(actual_pops):
            result.append(self._items.pop())
        
        self._pop_count += actual_pops
        return result
    
    def __len__(self) -> int:
        """Return current size."""
        return len(self._items)
    
    def stats(self) -> dict[str, Any]:
        """Return operation statistics."""
        total_ops = self._push_count + self._multipop_count
        return {
            "push_count": self._push_count,
            "pop_count": self._pop_count,
            "multipop_calls": self._multipop_count,
            "total_operations": total_ops,
            "amortised_pops_per_op": (
                self._pop_count / total_ops if total_ops > 0 else 0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BINARY COUNTER (POTENTIAL METHOD)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BinaryCounter:
    """
    A binary counter for demonstrating the potential method.
    
    The counter is represented as an array of bits. Incrementing may flip
    multiple bits, but the amortised cost is O(1).
    
    Potential Function:
        Φ(counter) = number of 1-bits in the counter
        
    Analysis:
        When incrementing:
        - Suppose we flip t bits from 1 to 0
        - Then flip 1 bit from 0 to 1
        - Actual cost: t + 1
        - Change in potential: 1 - t
        - Amortised cost: (t + 1) + (1 - t) = 2 = O(1)
    """
    
    _bits: list[int] = field(default_factory=lambda: [0] * 32)
    _flip_count: int = 0
    _increment_count: int = 0
    
    def increment(self) -> int:
        """
        Increment the counter by 1.
        
        Returns:
            Number of bit flips performed.
        """
        self._increment_count += 1
        flips = 0
        i = 0
        
        # Flip 1s to 0s until we find a 0
        while i < len(self._bits) and self._bits[i] == 1:
            self._bits[i] = 0
            flips += 1
            i += 1
        
        # Flip the first 0 to 1
        if i < len(self._bits):
            self._bits[i] = 1
            flips += 1
        
        self._flip_count += flips
        return flips
    
    def value(self) -> int:
        """Return current counter value."""
        result = 0
        for i, bit in enumerate(self._bits):
            if bit:
                result += 2 ** i
        return result
    
    def ones_count(self) -> int:
        """Return number of 1-bits (the potential)."""
        return sum(self._bits)
    
    def stats(self) -> dict[str, Any]:
        """Return counter statistics."""
        return {
            "value": self.value(),
            "increment_count": self._increment_count,
            "total_flips": self._flip_count,
            "amortised_flips_per_increment": (
                self._flip_count / self._increment_count 
                if self._increment_count > 0 else 0
            ),
            "current_potential": self.ones_count(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: YOUR TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_dynamic_array_growth(n: int) -> dict[str, Any]:
    """
    EXERCISE 1: Empirically verify amortised O(1) append.
    
    Create a DynamicArray and append n elements, tracking:
    - Total resize operations
    - Total element copies
    - Time per operation (running average)
    
    Args:
        n: Number of elements to append.
        
    Returns:
        Dictionary with:
        - "total_copies": Total copies made
        - "resize_count": Number of resizes
        - "copies_per_append": Average copies per append
        - "times": List of cumulative times at intervals
        
    Verify that copies_per_append stays close to 2 regardless of n.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement analyse_dynamic_array_growth")


def compare_growth_factors() -> dict[float, dict[str, float]]:
    """
    EXERCISE 2: Compare different growth factors.
    
    The classic growth factor is 2, but some implementations use 1.5.
    Compare growth factors of [1.25, 1.5, 2.0, 3.0] for n=100000 appends.
    
    Returns:
        Dictionary mapping growth_factor to:
        - "total_copies": Total copies made
        - "final_capacity": Final array capacity
        - "wasted_space": capacity - n (unused slots)
        - "copies_per_append": Amortised copies per append
        
    Question: What is the tradeoff between different growth factors?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compare_growth_factors")


def implement_shrinking_array() -> type:
    """
    EXERCISE 3: Implement a dynamic array that also shrinks.
    
    Create a class ShrinkingArray that:
    - Doubles capacity when full (like DynamicArray)
    - Halves capacity when less than 1/4 full
    
    This maintains O(1) amortised append AND remove from end.
    
    Returns:
        The ShrinkingArray class.
        
    Requirements:
        - Must have append() and pop() methods
        - Shrink when size < capacity / 4
        - Never shrink below minimum capacity of 4
        - Track resize statistics
        
    Why 1/4 and not 1/2?
        If we shrank at 1/2, alternating append/pop at the boundary
        would cause pathological resize behaviour.
    """
    # TODO: Implement and return the ShrinkingArray class
    raise NotImplementedError("Implement implement_shrinking_array")


def analyse_multipop_sequence(operations: list[tuple[str, int]]) -> dict[str, Any]:
    """
    EXERCISE 4: Analyse a sequence of stack operations.
    
    Given a sequence of operations, verify amortised analysis.
    
    Args:
        operations: List of (op_type, arg) tuples where:
            - ("push", _): Push a value (arg is the value)
            - ("pop", _): Pop one element (arg ignored)
            - ("multipop", k): Pop k elements
            
    Returns:
        Dictionary with:
        - "total_ops": Total operations (push + multipop calls)
        - "total_pops": Total elements popped
        - "max_multipop": Largest single multipop
        - "amortised_pops": Average pops per operation
        
    Verify that amortised_pops ≤ 2 regardless of multipop sizes.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement analyse_multipop_sequence")


def implement_increment_with_tracking() -> type:
    """
    EXERCISE 5: Implement a binary counter with potential tracking.
    
    Create a class TrackedCounter that:
    - Increments like BinaryCounter
    - Tracks potential (number of 1-bits) before and after each increment
    - Records amortised cost using: actual_cost + ΔΦ
    
    Returns:
        The TrackedCounter class.
        
    Requirements:
        - increment() method returns dict with:
            - "actual_cost": Number of bit flips
            - "potential_before": Φ before increment
            - "potential_after": Φ after increment
            - "amortised_cost": actual_cost + (after - before)
        - history() method returns list of all increment records
        - verify() method confirms all amortised costs ≤ 2
    """
    # TODO: Implement and return the TrackedCounter class
    raise NotImplementedError("Implement implement_increment_with_tracking")


def design_amortised_queue() -> type:
    """
    EXERCISE 6: Implement a queue with two stacks.
    
    Create a class TwoStackQueue that implements a queue using two stacks.
    
    Approach:
        - Use an "inbox" stack for enqueue
        - Use an "outbox" stack for dequeue
        - When outbox empty, transfer all from inbox to outbox
        
    Returns:
        The TwoStackQueue class.
        
    Requirements:
        - enqueue(x): O(1) always
        - dequeue(): O(1) amortised (O(n) worst case for transfer)
        - Track operation counts
        - Provide amortised analysis evidence
        
    Amortised Analysis:
        Each element is pushed at most twice (once to inbox, once to outbox)
        and popped at most twice. Total work for n operations ≤ 4n.
        Amortised cost per operation: O(1).
    """
    # TODO: Implement and return the TwoStackQueue class
    raise NotImplementedError("Implement design_amortised_queue")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BONUS CHALLENGES
# ═══════════════════════════════════════════════════════════════════════════════

def implement_hash_table_analysis() -> type:
    """
    BONUS 1: Analyse hash table with dynamic resizing.
    
    Implement a hash table that:
    - Doubles when load factor > 0.75
    - Halves when load factor < 0.25 (min size 8)
    
    Track and verify amortised O(1) operations.
    
    Returns:
        The DynamicHashTable class.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement implement_hash_table_analysis")


def compare_amortisation_methods() -> dict[str, str]:
    """
    BONUS 2: Write proofs for all three amortisation methods.
    
    For the dynamic array, provide formal proofs using:
    1. Aggregate method
    2. Accounting method
    3. Potential method
    
    Returns:
        Dictionary mapping method name to LaTeX-formatted proof string.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement compare_amortisation_methods")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: VERIFICATION AND DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def verify_implementations() -> bool:
    """Verify that implementations are correct."""
    # Test DynamicArray
    arr = DynamicArray[int]()
    for i in range(100):
        arr.append(i)
    assert len(arr) == 100
    assert arr.get(50) == 50
    stats = arr.stats()
    assert stats["amortised_copies_per_append"] < 3  # Should be close to 2
    
    # Test MultipopStack
    stack = MultipopStack[int]()
    for i in range(50):
        stack.push(i)
    popped = stack.multipop(30)
    assert len(popped) == 30
    assert len(stack) == 20
    
    # Test BinaryCounter
    counter = BinaryCounter()
    for _ in range(100):
        counter.increment()
    assert counter.value() == 100
    stats = counter.stats()
    assert stats["amortised_flips_per_increment"] < 3
    
    logger.info("All verifications passed!")
    return True


def demo() -> None:
    """Demonstrate amortised analysis concepts."""
    logger.info("=" * 70)
    logger.info("AMORTISED ANALYSIS DEMONSTRATION")
    logger.info("=" * 70)
    
    # Demo 1: Dynamic Array
    logger.info("\nDemo 1: Dynamic Array growth analysis")
    arr = DynamicArray[int]()
    
    checkpoints = [100, 1000, 10000, 100000]
    for target in checkpoints:
        while len(arr) < target:
            arr.append(len(arr))
        stats = arr.stats()
        logger.info(
            f"  n={target:>6}: resizes={stats['resize_count']:>2}, "
            f"copies/append={stats['amortised_copies_per_append']:.3f}"
        )
    
    # Demo 2: Binary Counter
    logger.info("\nDemo 2: Binary Counter potential analysis")
    
    for n in [100, 1000, 10000]:
        counter = BinaryCounter()
        for _ in range(n):
            counter.increment()
        stats = counter.stats()
        logger.info(
            f"  n={n:>5}: total_flips={stats['total_flips']:>5}, "
            f"flips/inc={stats['amortised_flips_per_increment']:.3f}"
        )
    
    # Demo 3: Multipop Stack
    logger.info("\nDemo 3: Multipop Stack accounting analysis")
    
    stack = MultipopStack[int]()
    for i in range(1000):
        stack.push(i)
    
    # Do some multipops
    while len(stack) > 0:
        k = min(len(stack), 100)
        stack.multipop(k)
    
    stats = stack.stats()
    logger.info(
        f"  pushes={stats['push_count']}, pops={stats['pop_count']}, "
        f"multipop_calls={stats['multipop_calls']}"
    )
    logger.info(
        f"  amortised pops per operation: {stats['amortised_pops_per_op']:.3f}"
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Complete exercises to master amortised analysis!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Amortised Analysis Practice Exercise"
    )
    parser.add_argument("--verify", action="store_true", help="Run verifications")
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    args = parser.parse_args()
    
    if args.verify:
        verify_implementations()
    elif args.demo:
        demo()
    else:
        logger.info("Use --verify to test or --demo to see demonstrations")


if __name__ == "__main__":
    main()
