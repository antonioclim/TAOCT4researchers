#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Solutions for hard_01_amortised_analysis.py
═══════════════════════════════════════════════════════════════════════════════

Complete solutions demonstrating amortised analysis techniques:
- Aggregate method
- Accounting method
- Potential method

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES FOR AMORTISED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DynamicArrayStats:
    """Statistics for dynamic array operations."""

    total_appends: int = 0
    total_resizes: int = 0
    total_copies: int = 0
    resize_history: list[tuple[int, int]] = field(default_factory=list)


class DynamicArray(Generic[T]):
    """Dynamic array with configurable growth factor.

    This implementation tracks all operations for amortised analysis.

    Amortised Analysis (Aggregate Method):
    - n appends cause at most log_g(n) resizes where g is growth_factor
    - Total copies: 1 + g + g² + ... + g^k = (g^(k+1) - 1)/(g-1) < 2n for g=2
    - Amortised cost per append: O(1)

    Attributes:
        growth_factor: Factor by which capacity increases on resize.
        stats: Statistics about operations performed.
    """

    def __init__(
        self, initial_capacity: int = 1, growth_factor: float = 2.0
    ) -> None:
        """Initialise dynamic array.

        Args:
            initial_capacity: Initial array capacity.
            growth_factor: Capacity multiplier on resize (must be > 1).
        """
        if growth_factor <= 1.0:
            raise ValueError("Growth factor must be greater than 1")

        self._data: list[T | None] = [None] * initial_capacity
        self._size: int = 0
        self._capacity: int = initial_capacity
        self.growth_factor: float = growth_factor
        self.stats: DynamicArrayStats = DynamicArrayStats()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> T:
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size})")
        return self._data[index]  # type: ignore[return-value]

    def append(self, value: T) -> int:
        """Append value to array, resizing if necessary.

        Args:
            value: Value to append.

        Returns:
            Cost of this operation (1 + copies if resize occurred).
        """
        cost = 1  # Base cost for the append

        if self._size >= self._capacity:
            # Need to resize
            old_capacity = self._capacity
            new_capacity = max(1, int(self._capacity * self.growth_factor))

            # Create new array and copy elements
            new_data: list[T | None] = [None] * new_capacity
            for i in range(self._size):
                new_data[i] = self._data[i]

            self._data = new_data
            self._capacity = new_capacity

            # Track statistics
            self.stats.total_resizes += 1
            self.stats.total_copies += self._size
            self.stats.resize_history.append((old_capacity, new_capacity))
            cost += self._size  # Copying cost

        self._data[self._size] = value
        self._size += 1
        self.stats.total_appends += 1

        return cost

    @property
    def capacity(self) -> int:
        """Current capacity of the array."""
        return self._capacity


class MultipopStack(Generic[T]):
    """Stack supporting multipop operation.

    Amortised Analysis (Accounting Method):
    - Assign $2 to each push: $1 pays for push, $1 saved for future pop
    - Each pop (including within multipop) uses $1 from savings
    - We can never pop more elements than we've pushed
    - Therefore: amortised cost of multipop is O(1) per element
    """

    def __init__(self) -> None:
        self._stack: list[T] = []
        self._push_count: int = 0
        self._pop_count: int = 0
        self._multipop_calls: int = 0

    def push(self, value: T) -> int:
        """Push value onto stack.

        Accounting: Pay $2 (1 for push + 1 credit for future pop).

        Returns:
            Actual cost: 1
        """
        self._stack.append(value)
        self._push_count += 1
        return 1

    def pop(self) -> T | None:
        """Pop single element from stack.

        Returns:
            Popped element or None if empty.
        """
        if not self._stack:
            return None
        self._pop_count += 1
        return self._stack.pop()

    def multipop(self, k: int) -> list[T]:
        """Pop up to k elements from stack.

        Accounting: Uses k credits previously deposited by push.

        Args:
            k: Maximum number of elements to pop.

        Returns:
            List of popped elements (may be fewer than k if stack smaller).
        """
        self._multipop_calls += 1
        result: list[T] = []
        actual_pops = min(k, len(self._stack))
        for _ in range(actual_pops):
            result.append(self._stack.pop())
            self._pop_count += 1
        return result

    def __len__(self) -> int:
        return len(self._stack)

    def get_stats(self) -> dict[str, int]:
        """Return operation statistics."""
        return {
            "push_count": self._push_count,
            "pop_count": self._pop_count,
            "multipop_calls": self._multipop_calls,
        }


class BinaryCounter:
    """Binary counter supporting increment operation.

    Amortised Analysis (Potential Method):
    - Potential function Φ = number of 1-bits
    - Increment flips trailing 1s to 0s, then one 0 to 1
    - Actual cost = (flips) = t + 1 where t = trailing 1s
    - ΔΦ = 1 - t (we add one 1-bit, remove t 1-bits)
    - Amortised cost = actual + ΔΦ = (t + 1) + (1 - t) = 2 = O(1)
    """

    def __init__(self, bits: int = 32) -> None:
        """Initialise binary counter.

        Args:
            bits: Number of bits in counter.
        """
        self._bits: list[int] = [0] * bits
        self._increment_count: int = 0
        self._total_flips: int = 0

    def increment(self) -> int:
        """Increment the counter by 1.

        Returns:
            Number of bit flips performed.
        """
        self._increment_count += 1
        flips = 0
        i = 0

        # Flip trailing 1s to 0s
        while i < len(self._bits) and self._bits[i] == 1:
            self._bits[i] = 0
            flips += 1
            i += 1

        # Flip the first 0 to 1
        if i < len(self._bits):
            self._bits[i] = 1
            flips += 1

        self._total_flips += flips
        return flips

    def value(self) -> int:
        """Return current counter value."""
        result = 0
        for i, bit in enumerate(self._bits):
            if bit:
                result += 1 << i
        return result

    def potential(self) -> int:
        """Return current potential (number of 1-bits)."""
        return sum(self._bits)

    def get_stats(self) -> dict[str, Any]:
        """Return counter statistics."""
        return {
            "value": self.value(),
            "potential": self.potential(),
            "increments": self._increment_count,
            "total_flips": self._total_flips,
            "average_flips_per_increment": (
                self._total_flips / self._increment_count
                if self._increment_count > 0
                else 0
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 1: ANALYSE DYNAMIC ARRAY GROWTH
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_dynamic_array_growth(
    n: int, growth_factor: float = 2.0
) -> dict[str, Any]:
    """Analyse the amortised cost of dynamic array appends.

    Uses the aggregate method to show total cost is O(n).

    Args:
        n: Number of appends to perform.
        growth_factor: Array growth factor.

    Returns:
        Analysis results including total cost and amortised cost.
    """
    array: DynamicArray[int] = DynamicArray(initial_capacity=1, growth_factor=growth_factor)
    total_cost = 0

    for i in range(n):
        cost = array.append(i)
        total_cost += cost

    # Theoretical analysis
    # Number of resizes: log_g(n) where g = growth_factor
    expected_resizes = int(math.log(n, growth_factor)) if n > 1 else 0

    # Total copies: sum of geometric series
    # 1 + g + g² + ... + g^k ≈ (g^(k+1) - 1)/(g - 1)
    k = array.stats.total_resizes
    if k > 0:
        theoretical_copies = int((growth_factor ** (k + 1) - 1) / (growth_factor - 1))
    else:
        theoretical_copies = 0

    results = {
        "n": n,
        "growth_factor": growth_factor,
        "total_cost": total_cost,
        "amortised_cost_per_append": total_cost / n,
        "total_resizes": array.stats.total_resizes,
        "expected_resizes_approx": expected_resizes,
        "total_copies": array.stats.total_copies,
        "theoretical_copies_approx": theoretical_copies,
        "final_capacity": array.capacity,
        "resize_history": array.stats.resize_history,
        "conclusion": (
            f"Amortised cost is {total_cost/n:.4f} per append, "
            f"confirming O(1) amortised time."
        ),
    }

    logger.info(
        "Dynamic array (n=%d, g=%.1f): amortised=%.4f, resizes=%d, copies=%d",
        n,
        growth_factor,
        total_cost / n,
        array.stats.total_resizes,
        array.stats.total_copies,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 2: COMPARE GROWTH FACTORS
# ═══════════════════════════════════════════════════════════════════════════════


def compare_growth_factors(n: int, factors: list[float]) -> dict[str, list[Any]]:
    """Compare amortised costs for different growth factors.

    Trade-off: larger factor = fewer resizes but more wasted space.

    Args:
        n: Number of operations.
        factors: List of growth factors to compare.

    Returns:
        Comparison results for each factor.
    """
    results: dict[str, list[Any]] = {
        "factor": [],
        "total_cost": [],
        "amortised_cost": [],
        "resizes": [],
        "copies": [],
        "final_capacity": [],
        "wasted_space": [],
    }

    for factor in factors:
        array: DynamicArray[int] = DynamicArray(initial_capacity=1, growth_factor=factor)
        total_cost = 0

        for i in range(n):
            total_cost += array.append(i)

        wasted = array.capacity - len(array)

        results["factor"].append(factor)
        results["total_cost"].append(total_cost)
        results["amortised_cost"].append(total_cost / n)
        results["resizes"].append(array.stats.total_resizes)
        results["copies"].append(array.stats.total_copies)
        results["final_capacity"].append(array.capacity)
        results["wasted_space"].append(wasted)

        logger.info(
            "Factor %.2f: amortised=%.4f, resizes=%d, wasted=%d",
            factor,
            total_cost / n,
            array.stats.total_resizes,
            wasted,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 3: IMPLEMENT SHRINKING ARRAY
# ═══════════════════════════════════════════════════════════════════════════════


class ShrinkingDynamicArray(Generic[T]):
    """Dynamic array that also shrinks when elements are removed.

    Key insight: Shrink threshold must be less than 1/growth_factor
    to avoid thrashing (repeated grow-shrink cycles).

    For growth_factor=2, we shrink when size < capacity/4 (not capacity/2).
    This maintains amortised O(1) for both insert and delete.
    """

    def __init__(
        self, initial_capacity: int = 4, growth_factor: float = 2.0
    ) -> None:
        if growth_factor <= 1.0:
            raise ValueError("Growth factor must be greater than 1")

        self._data: list[T | None] = [None] * initial_capacity
        self._size: int = 0
        self._capacity: int = initial_capacity
        self.growth_factor: float = growth_factor
        # Shrink when size < capacity * shrink_threshold
        # Must be < 1/growth_factor to avoid thrashing
        self.shrink_threshold: float = 1.0 / (growth_factor * growth_factor)
        self.min_capacity: int = 4

        self.total_grows: int = 0
        self.total_shrinks: int = 0
        self.total_copies: int = 0

    def _resize(self, new_capacity: int) -> None:
        """Resize internal array."""
        new_data: list[T | None] = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
            self.total_copies += 1
        self._data = new_data
        self._capacity = new_capacity

    def append(self, value: T) -> None:
        """Append value, growing if necessary."""
        if self._size >= self._capacity:
            new_cap = int(self._capacity * self.growth_factor)
            self._resize(new_cap)
            self.total_grows += 1
        self._data[self._size] = value
        self._size += 1

    def pop(self) -> T:
        """Remove and return last element, shrinking if needed.

        Returns:
            The removed element.

        Raises:
            IndexError: If array is empty.
        """
        if self._size == 0:
            raise IndexError("Pop from empty array")

        self._size -= 1
        value = self._data[self._size]
        self._data[self._size] = None

        # Shrink if size drops below threshold
        if (
            self._size < self._capacity * self.shrink_threshold
            and self._capacity > self.min_capacity
        ):
            new_cap = max(self.min_capacity, self._capacity // 2)
            self._resize(new_cap)
            self.total_shrinks += 1

        return value  # type: ignore[return-value]

    def __len__(self) -> int:
        return self._size

    def get_stats(self) -> dict[str, Any]:
        return {
            "size": self._size,
            "capacity": self._capacity,
            "grows": self.total_grows,
            "shrinks": self.total_shrinks,
            "total_copies": self.total_copies,
        }


def implement_shrinking_array(
    operations: list[tuple[str, int | None]]
) -> dict[str, Any]:
    """Demonstrate shrinking array with mixed operations.

    Args:
        operations: List of ('append', value) or ('pop', None) tuples.

    Returns:
        Statistics about the operations.
    """
    array: ShrinkingDynamicArray[int] = ShrinkingDynamicArray()
    op_costs: list[int] = []
    size_history: list[int] = []
    capacity_history: list[int] = []

    for op, value in operations:
        copies_before = array.total_copies
        if op == "append" and value is not None:
            array.append(value)
        elif op == "pop":
            try:
                array.pop()
            except IndexError:
                pass

        cost = 1 + (array.total_copies - copies_before)
        op_costs.append(cost)
        size_history.append(len(array))
        capacity_history.append(array._capacity)

    total_cost = sum(op_costs)
    n_ops = len(operations)

    stats = array.get_stats()
    stats["total_operations"] = n_ops
    stats["total_cost"] = total_cost
    stats["amortised_cost"] = total_cost / n_ops if n_ops > 0 else 0

    logger.info(
        "Shrinking array: %d ops, amortised=%.4f, grows=%d, shrinks=%d",
        n_ops,
        stats["amortised_cost"],
        stats["grows"],
        stats["shrinks"],
    )

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 4: ANALYSE MULTIPOP SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_multipop_sequence(
    operations: list[tuple[str, int]]
) -> dict[str, Any]:
    """Analyse amortised cost of multipop stack operations.

    Using the accounting method:
    - Each push deposits $2 (1 for push, 1 for future pop)
    - Each pop withdraws $1
    - Multipop(k) withdraws at most k dollars

    Args:
        operations: List of ('push', value) or ('multipop', k) tuples.

    Returns:
        Analysis of costs.
    """
    stack: MultipopStack[int] = MultipopStack()
    total_actual_cost = 0
    credits_deposited = 0
    credits_used = 0

    for op, arg in operations:
        if op == "push":
            actual_cost = stack.push(arg)
            total_actual_cost += actual_cost
            credits_deposited += 2  # Accounting: deposit $2
        elif op == "multipop":
            popped = stack.multipop(arg)
            actual_cost = len(popped)
            total_actual_cost += actual_cost
            credits_used += actual_cost  # Each pop uses $1

    n_ops = len(operations)
    stats = stack.get_stats()

    results = {
        "operations": n_ops,
        "pushes": stats["push_count"],
        "pops": stats["pop_count"],
        "multipop_calls": stats["multipop_calls"],
        "total_actual_cost": total_actual_cost,
        "amortised_cost_per_op": total_actual_cost / n_ops if n_ops > 0 else 0,
        "credits_deposited": credits_deposited,
        "credits_used": credits_used,
        "credits_remaining": credits_deposited - credits_used,
        "accounting_invariant": "Credits remaining >= 0: {}".format(
            credits_deposited >= credits_used
        ),
    }

    logger.info(
        "Multipop analysis: %d ops, actual_cost=%d, amortised=%.4f",
        n_ops,
        total_actual_cost,
        results["amortised_cost_per_op"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 5: INCREMENT WITH TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


def implement_increment_with_tracking(n: int) -> dict[str, Any]:
    """Track binary counter increments using potential method.

    Potential function: Φ(counter) = number of 1-bits

    Amortised cost = actual cost + ΔΦ
    For increment that flips t 1-bits to 0 and one 0 to 1:
    - Actual cost = t + 1
    - ΔΦ = 1 - t
    - Amortised = (t + 1) + (1 - t) = 2 = O(1)

    Args:
        n: Number of increments.

    Returns:
        Detailed tracking of potential method.
    """
    counter = BinaryCounter(bits=32)
    potential_history: list[int] = [counter.potential()]
    cost_history: list[int] = []
    amortised_history: list[float] = []

    total_actual = 0

    for i in range(n):
        phi_before = counter.potential()
        actual_cost = counter.increment()
        phi_after = counter.potential()

        delta_phi = phi_after - phi_before
        amortised = actual_cost + delta_phi

        total_actual += actual_cost
        cost_history.append(actual_cost)
        potential_history.append(phi_after)
        amortised_history.append(amortised)

    stats = counter.get_stats()

    results = {
        "n_increments": n,
        "final_value": stats["value"],
        "final_potential": stats["potential"],
        "total_actual_cost": total_actual,
        "total_flips": stats["total_flips"],
        "average_actual_cost": total_actual / n if n > 0 else 0,
        "average_amortised_cost": sum(amortised_history) / n if n > 0 else 0,
        "max_actual_cost": max(cost_history) if cost_history else 0,
        "amortised_bound": 2,  # Theoretical O(1) with constant 2
        "potential_method_works": all(a <= 3 for a in amortised_history),
    }

    logger.info(
        "Counter increments: n=%d, total_flips=%d, avg_actual=%.4f, avg_amortised=%.4f",
        n,
        total_actual,
        results["average_actual_cost"],
        results["average_amortised_cost"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 6: AMORTISED QUEUE
# ═══════════════════════════════════════════════════════════════════════════════


class AmortisedQueue(Generic[T]):
    """Queue implemented with two stacks, amortised O(1) operations.

    Uses the potential method:
    - Φ = size of input stack
    - Enqueue: actual=1, ΔΦ=+1, amortised=2
    - Dequeue when output empty: actual=k, ΔΦ=-k, amortised=0
    - Dequeue when output non-empty: actual=1, ΔΦ=0, amortised=1

    All operations are amortised O(1).
    """

    def __init__(self) -> None:
        self._input: list[T] = []   # For enqueue
        self._output: list[T] = []  # For dequeue
        self._enqueue_count: int = 0
        self._dequeue_count: int = 0
        self._transfers: int = 0
        self._total_transfer_cost: int = 0

    def enqueue(self, value: T) -> int:
        """Add element to queue.

        Returns:
            Actual cost (always 1).
        """
        self._input.append(value)
        self._enqueue_count += 1
        return 1

    def dequeue(self) -> tuple[T | None, int]:
        """Remove and return front element.

        Returns:
            Tuple of (element or None, actual cost).
        """
        if not self._output:
            if not self._input:
                return None, 1

            # Transfer all from input to output
            transfer_cost = len(self._input)
            while self._input:
                self._output.append(self._input.pop())
            self._transfers += 1
            self._total_transfer_cost += transfer_cost
            cost = transfer_cost + 1
        else:
            cost = 1

        self._dequeue_count += 1
        return self._output.pop(), cost

    def potential(self) -> int:
        """Return current potential (size of input stack)."""
        return len(self._input)

    def __len__(self) -> int:
        return len(self._input) + len(self._output)

    def get_stats(self) -> dict[str, Any]:
        return {
            "enqueues": self._enqueue_count,
            "dequeues": self._dequeue_count,
            "transfers": self._transfers,
            "total_transfer_cost": self._total_transfer_cost,
            "current_potential": self.potential(),
        }


def design_amortised_queue(operations: list[tuple[str, int | None]]) -> dict[str, Any]:
    """Demonstrate amortised queue analysis.

    Args:
        operations: List of ('enqueue', value) or ('dequeue', None) tuples.

    Returns:
        Analysis results.
    """
    queue: AmortisedQueue[int] = AmortisedQueue()
    total_actual_cost = 0
    potential_history: list[int] = [0]
    amortised_costs: list[float] = []

    for op, value in operations:
        phi_before = queue.potential()

        if op == "enqueue" and value is not None:
            cost = queue.enqueue(value)
        else:  # dequeue
            _, cost = queue.dequeue()

        phi_after = queue.potential()
        delta_phi = phi_after - phi_before
        amortised = cost + delta_phi

        total_actual_cost += cost
        potential_history.append(phi_after)
        amortised_costs.append(amortised)

    n_ops = len(operations)
    stats = queue.get_stats()

    results = {
        "operations": n_ops,
        "enqueues": stats["enqueues"],
        "dequeues": stats["dequeues"],
        "transfers": stats["transfers"],
        "total_actual_cost": total_actual_cost,
        "amortised_cost_per_op": total_actual_cost / n_ops if n_ops > 0 else 0,
        "max_amortised_cost": max(amortised_costs) if amortised_costs else 0,
        "amortised_bound_holds": all(c <= 2 for c in amortised_costs),
        "final_potential": stats["current_potential"],
    }

    logger.info(
        "Amortised queue: %d ops, transfers=%d, amortised=%.4f",
        n_ops,
        stats["transfers"],
        results["amortised_cost_per_op"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: HASH TABLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def implement_hash_table_analysis(n: int, load_threshold: float = 0.75) -> dict[str, Any]:
    """Analyse amortised cost of hash table with resizing.

    Similar to dynamic array: resize when load factor exceeds threshold.
    Amortised O(1) insert assuming good hash function.

    Args:
        n: Number of insertions.
        load_threshold: Load factor threshold for resizing.

    Returns:
        Analysis results.
    """
    capacity = 4
    size = 0
    total_cost = 0
    resize_count = 0
    total_rehashes = 0

    for i in range(n):
        cost = 1  # Base insert cost

        # Check if resize needed
        if size / capacity >= load_threshold:
            # Double capacity, rehash all elements
            rehash_cost = size
            total_cost += rehash_cost
            total_rehashes += size
            capacity *= 2
            resize_count += 1
            cost += rehash_cost

        size += 1
        total_cost += 1

    results = {
        "insertions": n,
        "final_size": size,
        "final_capacity": capacity,
        "load_factor": size / capacity,
        "resize_count": resize_count,
        "total_rehashes": total_rehashes,
        "total_cost": total_cost,
        "amortised_cost": total_cost / n,
        "conclusion": "Amortised O(1) insertion confirmed.",
    }

    logger.info(
        "Hash table: n=%d, resizes=%d, rehashes=%d, amortised=%.4f",
        n,
        resize_count,
        total_rehashes,
        results["amortised_cost"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: COMPARE AMORTISATION METHODS
# ═══════════════════════════════════════════════════════════════════════════════


def compare_amortisation_methods(n: int = 1000) -> dict[str, Any]:
    """Compare the three amortisation methods on the same problem.

    Problem: Dynamic array analysis

    Args:
        n: Number of operations.

    Returns:
        Comparison of methods.
    """
    # Perform actual operations
    array: DynamicArray[int] = DynamicArray(growth_factor=2.0)
    actual_costs: list[int] = []

    for i in range(n):
        cost = array.append(i)
        actual_costs.append(cost)

    total_actual = sum(actual_costs)

    # Method 1: Aggregate
    aggregate_amortised = total_actual / n

    # Method 2: Accounting
    # Charge $3 per operation: $1 actual + $2 credit
    accounting_charge = 3
    credits_needed = total_actual  # All work must be paid for
    credits_available = n * accounting_charge
    accounting_valid = credits_available >= credits_needed

    # Method 3: Potential
    # Φ = 2 * size - capacity (measures "pressure" to resize)
    potential_costs: list[float] = []
    phi = 0  # Initial potential
    size = 0
    cap = 1

    for cost in actual_costs:
        old_phi = 2 * size - cap
        size += 1
        if cost > 1:  # Resize occurred
            cap = int(cap * 2)
        new_phi = 2 * size - cap
        amortised = cost + (new_phi - old_phi)
        potential_costs.append(amortised)

    potential_max = max(potential_costs)
    potential_avg = sum(potential_costs) / len(potential_costs)

    results = {
        "n": n,
        "total_actual_cost": total_actual,
        "aggregate_method": {
            "amortised_per_op": aggregate_amortised,
            "bound": "O(1)",
        },
        "accounting_method": {
            "charge_per_op": accounting_charge,
            "credits_available": credits_available,
            "credits_needed": credits_needed,
            "valid": accounting_valid,
            "bound": "O(1)" if accounting_valid else "Invalid",
        },
        "potential_method": {
            "average_amortised": potential_avg,
            "max_amortised": potential_max,
            "potential_function": "Φ = 2·size - capacity",
            "bound": "O(1)",
        },
        "conclusion": "All three methods confirm O(1) amortised cost per append.",
    }

    logger.info(
        "Comparison: aggregate=%.4f, accounting=%d, potential_avg=%.4f",
        aggregate_amortised,
        accounting_charge,
        potential_avg,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_demonstrations() -> None:
    """Run all amortised analysis demonstrations."""
    logger.info("=" * 70)
    logger.info("AMORTISED ANALYSIS SOLUTIONS DEMONSTRATION")
    logger.info("=" * 70)

    # Demo 1: Dynamic array growth
    logger.info("\n--- Dynamic Array Growth Analysis ---")
    analyse_dynamic_array_growth(10000, growth_factor=2.0)

    # Demo 2: Compare growth factors
    logger.info("\n--- Growth Factor Comparison ---")
    compare_growth_factors(10000, [1.5, 2.0, 3.0, 4.0])

    # Demo 3: Shrinking array
    logger.info("\n--- Shrinking Array ---")
    ops: list[tuple[str, int | None]] = []
    for i in range(1000):
        ops.append(("append", i))
    for _ in range(800):
        ops.append(("pop", None))
    for i in range(500):
        ops.append(("append", i))
    implement_shrinking_array(ops)

    # Demo 4: Multipop analysis
    logger.info("\n--- Multipop Stack Analysis ---")
    multipop_ops = [("push", i) for i in range(100)]
    multipop_ops.extend([("multipop", 10) for _ in range(5)])
    multipop_ops.extend([("push", i) for i in range(50)])
    multipop_ops.append(("multipop", 100))
    analyse_multipop_sequence(multipop_ops)

    # Demo 5: Binary counter
    logger.info("\n--- Binary Counter Analysis ---")
    implement_increment_with_tracking(1000)

    # Demo 6: Amortised queue
    logger.info("\n--- Amortised Queue ---")
    queue_ops: list[tuple[str, int | None]] = []
    for i in range(500):
        queue_ops.append(("enqueue", i))
    for _ in range(250):
        queue_ops.append(("dequeue", None))
    for i in range(250):
        queue_ops.append(("enqueue", i))
    for _ in range(500):
        queue_ops.append(("dequeue", None))
    design_amortised_queue(queue_ops)

    # Bonus: Method comparison
    logger.info("\n--- Amortisation Methods Comparison ---")
    compare_amortisation_methods(10000)

    logger.info("\n" + "=" * 70)
    logger.info("All demonstrations completed successfully")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demonstrations()
