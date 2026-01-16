#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: easy_03_loop_analysis.py
Week 3, Practice Exercise Solutions
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Analyse Loop Complexity
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_single_loop(n: int) -> str:
    """
    SOLUTION: Analyse complexity of a single loop.
    
    for i in range(n):    # Executes n times
        print(i)          # O(1) per iteration
    
    Total: O(n)
    """
    return "O(n) - The loop executes n times with O(1) work per iteration."


def analyse_nested_loops(n: int) -> str:
    """
    SOLUTION: Analyse complexity of nested loops.
    
    for i in range(n):        # Outer: n times
        for j in range(n):    # Inner: n times per outer
            print(i, j)       # O(1)
    
    Total: O(n²)
    """
    return "O(n²) - Outer loop runs n times, inner loop runs n times each = n×n"


def analyse_dependent_loops(n: int) -> str:
    """
    SOLUTION: Analyse loops where inner depends on outer.
    
    for i in range(n):
        for j in range(i):    # j goes from 0 to i-1
            print(i, j)
    
    Iterations: 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)
    """
    return "O(n²) - Sum of 0+1+2+...+(n-1) = n(n-1)/2 which is Θ(n²)"


def analyse_logarithmic_loop(n: int) -> str:
    """
    SOLUTION: Analyse loop with halving.
    
    i = n
    while i > 0:
        print(i)
        i = i // 2
    
    Iterations: log₂(n) since we halve each time
    """
    return "O(log n) - We halve i each iteration, so log₂(n) iterations"


def analyse_nested_log_linear(n: int) -> str:
    """
    SOLUTION: Analyse O(n log n) loop pattern.
    
    for i in range(n):        # n iterations
        j = 1
        while j < n:          # log(n) iterations
            print(i, j)
            j *= 2
    
    Total: O(n log n)
    """
    return "O(n log n) - Outer loop n times, inner loop log(n) times each"


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Count Operations
# ═══════════════════════════════════════════════════════════════════════════════

def count_operations_linear(n: int) -> int:
    """
    SOLUTION: Count exact operations in linear loop.
    """
    count = 0
    for i in range(n):
        count += 1  # This is the operation we're counting
    return count  # Returns n


def count_operations_quadratic(n: int) -> int:
    """
    SOLUTION: Count exact operations in nested loops.
    """
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count  # Returns n²


def count_operations_triangular(n: int) -> int:
    """
    SOLUTION: Count operations in triangular loop pattern.
    """
    count = 0
    for i in range(n):
        for j in range(i + 1):
            count += 1
    return count  # Returns n(n+1)/2


def count_operations_logarithmic(n: int) -> int:
    """
    SOLUTION: Count operations in logarithmic loop.
    """
    count = 0
    i = n
    while i > 0:
        count += 1
        i //= 2
    return count  # Returns floor(log₂(n)) + 1


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Predict Complexity
# ═══════════════════════════════════════════════════════════════════════════════

def predict_complexity(code_pattern: str) -> str:
    """
    SOLUTION: Given a code pattern description, predict complexity.
    """
    patterns = {
        "single_loop": "O(n)",
        "nested_equal": "O(n²)",
        "nested_dependent": "O(n²)",
        "triple_nested": "O(n³)",
        "halving": "O(log n)",
        "linear_times_log": "O(n log n)",
        "doubling_outer": "O(n)",  # 1+2+4+...+n ≈ 2n
    }
    return patterns.get(code_pattern, "Unknown pattern")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Empirical Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_complexity_empirically(
    func: callable,
    sizes: list[int]
) -> list[tuple[int, int, float]]:
    """
    SOLUTION: Run function and check operation count vs prediction.
    
    Returns list of (n, actual_count, ratio_to_previous).
    """
    results = []
    prev_count = None
    
    for n in sizes:
        count = func(n)
        ratio = count / prev_count if prev_count else 0
        results.append((n, count, ratio))
        prev_count = count
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo_solutions() -> None:
    """Demonstrate all solutions."""
    logger.info("=" * 60)
    logger.info("LOOP ANALYSIS SOLUTIONS DEMONSTRATION")
    logger.info("=" * 60)
    
    # Demo 1: Complexity analysis
    logger.info("\n1. Loop Complexity Analysis:")
    logger.info(f"   Single loop: {analyse_single_loop(100)}")
    logger.info(f"   Nested equal: {analyse_nested_loops(100)}")
    logger.info(f"   Nested dependent: {analyse_dependent_loops(100)}")
    logger.info(f"   Logarithmic: {analyse_logarithmic_loop(100)}")
    
    # Demo 2: Operation counting
    logger.info("\n2. Operation Counts:")
    for n in [10, 100, 1000]:
        linear = count_operations_linear(n)
        quad = count_operations_quadratic(n)
        tri = count_operations_triangular(n)
        log = count_operations_logarithmic(n)
        logger.info(f"   n={n}: linear={linear}, quad={quad}, tri={tri}, log={log}")
    
    # Demo 3: Verify quadratic growth
    logger.info("\n3. Verify O(n²) - ratio should approach 4 when n doubles:")
    results = verify_complexity_empirically(
        count_operations_quadratic,
        [100, 200, 400, 800, 1600]
    )
    for n, count, ratio in results:
        logger.info(f"   n={n}: count={count}, ratio={ratio:.2f}")
    
    # Demo 4: Verify logarithmic growth
    logger.info("\n4. Verify O(log n) - count should increase by ~1 when n doubles:")
    results = verify_complexity_empirically(
        count_operations_logarithmic,
        [100, 200, 400, 800, 1600]
    )
    for n, count, ratio in results:
        logger.info(f"   n={n}: count={count}")


if __name__ == "__main__":
    demo_solutions()
