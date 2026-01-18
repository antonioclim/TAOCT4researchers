#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Algorithmic Complexity - Complexity Proof Solutions
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for exercises on formal complexity analysis and proofs.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROOF DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BigOProof:
    """Formal Big-O proof with constants and explanation."""

    function_description: str
    big_o_class: str
    constant_c: float
    threshold_n0: int
    proof_steps: list[str]
    verification_values: list[tuple[int, float, float]]  # (n, f(n), c*g(n))


@dataclass
class BigThetaProof:
    """Formal Big-Θ proof with upper and lower bounds."""

    function_description: str
    big_theta_class: str
    constant_c1: float  # Lower bound constant
    constant_c2: float  # Upper bound constant
    threshold_n0: int
    proof_steps: list[str]
    verification_values: list[tuple[int, float, float, float]]  # (n, f(n), c1*g(n), c2*g(n))


@dataclass
class RecurrenceProof:
    """Proof for recurrence relation solution."""

    recurrence: str
    solution: str
    method: str  # "substitution", "master_theorem", "recursion_tree"
    proof_steps: list[str]
    verification_values: list[tuple[int, int]]  # (n, T(n))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXERCISE 1 SOLUTION - LINEAR SEARCH PROOF
# ═══════════════════════════════════════════════════════════════════════════════


def prove_linear_search() -> BigThetaProof:
    """
    SOLUTION: Prove that linear search is Θ(n) in the worst case.

    Theorem: Linear search has worst-case time complexity Θ(n).

    Proof Strategy:
    1. Count exact number of comparisons in worst case
    2. Show upper bound O(n): f(n) ≤ c₂·n for some c₂, n₀
    3. Show lower bound Ω(n): f(n) ≥ c₁·n for some c₁, n₀
    4. Therefore f(n) ∈ Θ(n)

    Returns:
        BigThetaProof with complete formal proof.
    """
    proof_steps = [
        "Let f(n) = number of comparisons in worst-case linear search on array of size n.",
        "",
        "WORST CASE ANALYSIS:",
        "The worst case occurs when the target is not in the array or is the last element.",
        "In this case, we must examine all n elements.",
        "Therefore, f(n) = n comparisons exactly.",
        "",
        "UPPER BOUND (O(n)):",
        "Claim: f(n) ∈ O(n)",
        "Proof: We need to find c₂ > 0 and n₀ ≥ 1 such that f(n) ≤ c₂·n for all n ≥ n₀.",
        "Since f(n) = n, we have f(n) = n ≤ 1·n for all n ≥ 1.",
        "Choose c₂ = 1, n₀ = 1. ✓",
        "",
        "LOWER BOUND (Ω(n)):",
        "Claim: f(n) ∈ Ω(n)",
        "Proof: We need to find c₁ > 0 and n₀ ≥ 1 such that f(n) ≥ c₁·n for all n ≥ n₀.",
        "Since f(n) = n, we have f(n) = n ≥ 1·n for all n ≥ 1.",
        "Choose c₁ = 1, n₀ = 1. ✓",
        "",
        "CONCLUSION:",
        "Since f(n) ∈ O(n) and f(n) ∈ Ω(n), we have f(n) ∈ Θ(n).",
        "Linear search has worst-case time complexity Θ(n). □",
    ]

    # Verification values
    verification = []
    for n in [1, 10, 100, 1000, 10000]:
        f_n = n  # Exact worst-case comparisons
        c1_g_n = 1 * n  # Lower bound
        c2_g_n = 1 * n  # Upper bound
        verification.append((n, f_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="f(n) = n (worst-case comparisons in linear search)",
        big_theta_class="Θ(n)",
        constant_c1=1.0,
        constant_c2=1.0,
        threshold_n0=1,
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXERCISE 2 SOLUTION - BUBBLE SORT PROOF
# ═══════════════════════════════════════════════════════════════════════════════


def prove_bubble_sort() -> BigThetaProof:
    """
    SOLUTION: Prove that bubble sort is Θ(n²) in the worst case.

    Theorem: Bubble sort has worst-case time complexity Θ(n²).

    Proof Strategy:
    1. Count exact comparisons in worst case (reverse-sorted input)
    2. Show the count is n(n-1)/2
    3. Prove this is Θ(n²)

    Returns:
        BigThetaProof with complete formal proof.
    """
    proof_steps = [
        "Let f(n) = number of comparisons in worst-case bubble sort on array of size n.",
        "",
        "WORST CASE ANALYSIS:",
        "The worst case occurs when the array is reverse-sorted.",
        "In each pass i (0 to n-2), we make (n-1-i) comparisons.",
        "",
        "Total comparisons:",
        "f(n) = Σᵢ₌₀ⁿ⁻² (n-1-i) = (n-1) + (n-2) + ... + 1 + 0",
        "     = Σⱼ₌₀ⁿ⁻¹ j = (n-1)n/2 = n²/2 - n/2",
        "",
        "UPPER BOUND (O(n²)):",
        "Claim: f(n) = n²/2 - n/2 ∈ O(n²)",
        "For n ≥ 1: f(n) = n²/2 - n/2 ≤ n²/2 ≤ n²",
        "Choose c₂ = 1, n₀ = 1. ✓",
        "",
        "LOWER BOUND (Ω(n²)):",
        "Claim: f(n) = n²/2 - n/2 ∈ Ω(n²)",
        "For n ≥ 2: f(n) = n²/2 - n/2 = n²/2(1 - 1/n) ≥ n²/2 · (1/2) = n²/4",
        "Choose c₁ = 1/4, n₀ = 2. ✓",
        "",
        "CONCLUSION:",
        "Since f(n) ∈ O(n²) and f(n) ∈ Ω(n²), we have f(n) ∈ Θ(n²).",
        "Bubble sort has worst-case time complexity Θ(n²). □",
    ]

    # Verification values
    verification = []
    for n in [2, 5, 10, 50, 100]:
        f_n = n * (n - 1) / 2  # Exact comparisons
        c1_g_n = 0.25 * n * n  # Lower bound
        c2_g_n = 1.0 * n * n  # Upper bound
        verification.append((n, f_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="f(n) = n(n-1)/2 (worst-case comparisons in bubble sort)",
        big_theta_class="Θ(n²)",
        constant_c1=0.25,
        constant_c2=1.0,
        threshold_n0=2,
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EXERCISE 3 SOLUTION - BINARY SEARCH PROOF
# ═══════════════════════════════════════════════════════════════════════════════


def prove_binary_search() -> BigThetaProof:
    """
    SOLUTION: Prove that binary search is Θ(log n) in the worst case.

    Theorem: Binary search has worst-case time complexity Θ(log n).

    Proof Strategy:
    1. Show search space halves each iteration
    2. Count maximum iterations until search space is empty
    3. Prove this is Θ(log n)

    Returns:
        BigThetaProof with complete formal proof.
    """
    proof_steps = [
        "Let f(n) = number of comparisons in worst-case binary search on sorted array of size n.",
        "",
        "WORST CASE ANALYSIS:",
        "The worst case occurs when the target is not in the array.",
        "Each comparison halves the search space.",
        "",
        "After k comparisons, the search space has size at most n/2ᵏ.",
        "We stop when n/2ᵏ < 1, i.e., when 2ᵏ > n, i.e., when k > log₂(n).",
        "Therefore, f(n) = ⌊log₂(n)⌋ + 1 comparisons.",
        "",
        "For simplicity, we use f(n) ≈ log₂(n) in asymptotic analysis.",
        "",
        "UPPER BOUND (O(log n)):",
        "Claim: f(n) = ⌊log₂(n)⌋ + 1 ∈ O(log n)",
        "For n ≥ 2: f(n) = ⌊log₂(n)⌋ + 1 ≤ log₂(n) + 1 ≤ 2·log₂(n)",
        "Choose c₂ = 2, n₀ = 2. ✓",
        "",
        "LOWER BOUND (Ω(log n)):",
        "Claim: f(n) = ⌊log₂(n)⌋ + 1 ∈ Ω(log n)",
        "For n ≥ 2: f(n) = ⌊log₂(n)⌋ + 1 ≥ log₂(n) ≥ (1/2)·log₂(n) for n ≥ 2",
        "Actually, f(n) ≥ log₂(n) directly, so c₁ = 1 works.",
        "Choose c₁ = 1/2, n₀ = 2. ✓",
        "",
        "CONCLUSION:",
        "Since f(n) ∈ O(log n) and f(n) ∈ Ω(log n), we have f(n) ∈ Θ(log n).",
        "Binary search has worst-case time complexity Θ(log n). □",
    ]

    # Verification values
    verification = []
    for n in [2, 8, 64, 1024, 1048576]:
        f_n = math.floor(math.log2(n)) + 1  # Exact comparisons
        log_n = math.log2(n)
        c1_g_n = 0.5 * log_n  # Lower bound
        c2_g_n = 2.0 * log_n  # Upper bound
        verification.append((n, f_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="f(n) = ⌊log₂(n)⌋ + 1 (worst-case comparisons in binary search)",
        big_theta_class="Θ(log n)",
        constant_c1=0.5,
        constant_c2=2.0,
        threshold_n0=2,
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXERCISE 4 SOLUTION - MERGE SORT RECURRENCE
# ═══════════════════════════════════════════════════════════════════════════════


def prove_merge_sort_recurrence() -> RecurrenceProof:
    """
    SOLUTION: Prove merge sort complexity using the Master Theorem.

    Theorem: Merge sort has time complexity Θ(n log n).

    Recurrence: T(n) = 2T(n/2) + Θ(n)

    Proof Strategy:
    Apply the Master Theorem for recurrences of the form T(n) = aT(n/b) + f(n).

    Returns:
        RecurrenceProof with complete derivation.
    """
    proof_steps = [
        "MERGE SORT RECURRENCE:",
        "T(n) = 2T(n/2) + cn    where c is a constant for merging",
        "Base case: T(1) = c₀  (constant time for single element)",
        "",
        "APPLYING MASTER THEOREM:",
        "The Master Theorem applies to recurrences of the form:",
        "T(n) = aT(n/b) + f(n)",
        "",
        "For merge sort: a = 2, b = 2, f(n) = cn",
        "",
        "We compare f(n) with n^(log_b(a)) = n^(log₂(2)) = n¹ = n",
        "",
        "CASE 2 of Master Theorem:",
        "If f(n) = Θ(n^(log_b(a))) then T(n) = Θ(n^(log_b(a)) · log n)",
        "",
        "Since f(n) = cn = Θ(n) and n^(log₂(2)) = n, we have f(n) = Θ(n^(log_b(a))).",
        "Therefore, Case 2 applies.",
        "",
        "CONCLUSION:",
        "T(n) = Θ(n · log n)",
        "",
        "ALTERNATIVE: RECURSION TREE METHOD",
        "Level 0: cn work (1 problem of size n)",
        "Level 1: 2 · c(n/2) = cn work (2 problems of size n/2)",
        "Level 2: 4 · c(n/4) = cn work (4 problems of size n/4)",
        "...",
        "Level k: 2ᵏ · c(n/2ᵏ) = cn work (2ᵏ problems of size n/2ᵏ)",
        "",
        "Number of levels: log₂(n) + 1 (until n/2ᵏ = 1)",
        "Total work: cn · (log₂(n) + 1) = Θ(n log n). □",
    ]

    # Compute T(n) by simulation
    def merge_sort_count(n: int) -> int:
        """Count operations in merge sort recursively."""
        if n <= 1:
            return 1
        return 2 * merge_sort_count(n // 2) + n

    verification = []
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        t_n = merge_sort_count(n)
        verification.append((n, t_n))

    return RecurrenceProof(
        recurrence="T(n) = 2T(n/2) + cn",
        solution="T(n) = Θ(n log n)",
        method="master_theorem",
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXERCISE 5 SOLUTION - PROVE NOT IN BIG-O
# ═══════════════════════════════════════════════════════════════════════════════


def prove_not_in_big_o() -> BigOProof:
    """
    SOLUTION: Prove that n² is NOT in O(n).

    Theorem: n² ∉ O(n)

    Proof Strategy:
    Proof by contradiction - assume n² ∈ O(n) and derive a contradiction.

    Returns:
        BigOProof demonstrating the contradiction.
    """
    proof_steps = [
        "THEOREM: n² ∉ O(n)",
        "",
        "PROOF BY CONTRADICTION:",
        "Assume, for the sake of contradiction, that n² ∈ O(n).",
        "",
        "By definition of Big-O, there exist constants c > 0 and n₀ ≥ 1 such that:",
        "n² ≤ c·n for all n ≥ n₀",
        "",
        "Dividing both sides by n (valid for n > 0):",
        "n ≤ c for all n ≥ n₀",
        "",
        "But this is a contradiction!",
        "For any fixed constant c, we can choose n = c + 1, and then:",
        "n = c + 1 > c",
        "",
        "This contradicts the requirement that n ≤ c for all n ≥ n₀.",
        "",
        "Therefore, our assumption that n² ∈ O(n) must be false.",
        "",
        "CONCLUSION: n² ∉ O(n). □",
        "",
        "INTUITION:",
        "n² grows faster than n. No matter how large a constant c you choose,",
        "eventually n² will exceed c·n. This is because lim(n→∞) n²/(c·n) = lim(n→∞) n/c = ∞.",
    ]

    # Verification: show n² eventually exceeds any c·n
    verification = []
    c = 1000  # Use a large constant
    for n in [1, 10, 100, 1000, 10000]:
        f_n = n * n
        c_g_n = c * n
        verification.append((n, f_n, c_g_n))

    return BigOProof(
        function_description="f(n) = n²",
        big_o_class="NOT O(n)",
        constant_c=float("inf"),  # No valid c exists
        threshold_n0=-1,  # No valid n₀ exists
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EXERCISE 6 SOLUTION - SUM COMPLEXITY
# ═══════════════════════════════════════════════════════════════════════════════


def prove_sum_complexity() -> BigThetaProof:
    """
    SOLUTION: Prove that Σᵢ₌₁ⁿ i = n(n+1)/2 ∈ Θ(n²).

    Theorem: The sum 1 + 2 + ... + n = Θ(n²)

    This is a fundamental result used in analysing many algorithms
    with nested loops.

    Returns:
        BigThetaProof with complete derivation.
    """
    proof_steps = [
        "THEOREM: Σᵢ₌₁ⁿ i = n(n+1)/2 ∈ Θ(n²)",
        "",
        "First, we derive the closed form:",
        "Let S = 1 + 2 + 3 + ... + (n-1) + n",
        "Also S = n + (n-1) + (n-2) + ... + 2 + 1  (reversed)",
        "",
        "Adding: 2S = (1+n) + (2+n-1) + (3+n-2) + ... + (n+1)",
        "           = (n+1) + (n+1) + (n+1) + ... + (n+1)  (n terms)",
        "           = n(n+1)",
        "",
        "Therefore: S = n(n+1)/2",
        "",
        "Now we prove S = n(n+1)/2 ∈ Θ(n²):",
        "",
        "EXPANDING:",
        "f(n) = n(n+1)/2 = (n² + n)/2 = n²/2 + n/2",
        "",
        "UPPER BOUND (O(n²)):",
        "f(n) = n²/2 + n/2 ≤ n²/2 + n²/2 = n²  for n ≥ 1",
        "Choose c₂ = 1, n₀ = 1. ✓",
        "",
        "LOWER BOUND (Ω(n²)):",
        "f(n) = n²/2 + n/2 ≥ n²/2  for n ≥ 1",
        "Choose c₁ = 1/2, n₀ = 1. ✓",
        "",
        "Alternatively, for a tighter lower bound:",
        "f(n) = n²/2(1 + 1/n) ≥ n²/2 · 1 = n²/2",
        "",
        "CONCLUSION:",
        "Since f(n) ∈ O(n²) with c₂ = 1 and f(n) ∈ Ω(n²) with c₁ = 1/2,",
        "we have f(n) = n(n+1)/2 ∈ Θ(n²). □",
        "",
        "APPLICATION:",
        "This result explains why algorithms with nested loops where the inner",
        "loop runs 1, 2, 3, ..., n times have O(n²) complexity.",
    ]

    # Verification values
    verification = []
    for n in [1, 5, 10, 50, 100, 1000]:
        f_n = n * (n + 1) / 2
        c1_g_n = 0.5 * n * n
        c2_g_n = 1.0 * n * n
        verification.append((n, f_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="f(n) = n(n+1)/2 = Σᵢ₌₁ⁿ i",
        big_theta_class="Θ(n²)",
        constant_c1=0.5,
        constant_c2=1.0,
        threshold_n0=1,
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BONUS SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def prove_log_factorial() -> BigThetaProof:
    """
    BONUS SOLUTION: Prove that log(n!) ∈ Θ(n log n).

    Theorem: log(n!) = Θ(n log n)

    This is important for understanding information-theoretic lower bounds
    on comparison-based sorting.

    Returns:
        BigThetaProof with Stirling's approximation.
    """
    proof_steps = [
        "THEOREM: log(n!) ∈ Θ(n log n)",
        "",
        "We use Stirling's approximation: n! ≈ √(2πn)(n/e)ⁿ",
        "",
        "Taking logarithms:",
        "log(n!) = log(√(2πn)) + n·log(n/e)",
        "        = (1/2)log(2πn) + n·log(n) - n·log(e)",
        "        = (1/2)log(2π) + (1/2)log(n) + n·log(n) - n/ln(2)",
        "",
        "The dominant term is n·log(n).",
        "",
        "UPPER BOUND (O(n log n)):",
        "log(n!) = log(1) + log(2) + ... + log(n)",
        "        ≤ log(n) + log(n) + ... + log(n)  (n terms)",
        "        = n·log(n)",
        "Choose c₂ = 1, n₀ = 1. ✓",
        "",
        "LOWER BOUND (Ω(n log n)):",
        "log(n!) = log(1) + log(2) + ... + log(n)",
        "        ≥ log(n/2) + log(n/2+1) + ... + log(n)  (n/2 terms)",
        "        ≥ (n/2)·log(n/2)",
        "        = (n/2)·(log(n) - 1)",
        "        ≥ (n/4)·log(n)  for n ≥ 4",
        "Choose c₁ = 1/4, n₀ = 4. ✓",
        "",
        "CONCLUSION: log(n!) ∈ Θ(n log n). □",
        "",
        "SIGNIFICANCE:",
        "This proves that any comparison-based sorting algorithm requires",
        "Ω(n log n) comparisons, since there are n! possible orderings and",
        "each comparison provides at most 1 bit of information.",
    ]

    # Verification
    verification = []
    for n in [4, 8, 16, 32, 64, 128]:
        f_n = sum(math.log2(i) for i in range(1, n + 1))  # log(n!)
        log_n = math.log2(n)
        c1_g_n = 0.25 * n * log_n
        c2_g_n = 1.0 * n * log_n
        verification.append((n, f_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="f(n) = log₂(n!)",
        big_theta_class="Θ(n log n)",
        constant_c1=0.25,
        constant_c2=1.0,
        threshold_n0=4,
        proof_steps=proof_steps,
        verification_values=verification,
    )


def prove_harmonic_series() -> BigThetaProof:
    """
    BONUS SOLUTION: Prove that the harmonic series Hₙ = Σᵢ₌₁ⁿ 1/i ∈ Θ(log n).

    Theorem: Hₙ = 1 + 1/2 + 1/3 + ... + 1/n = Θ(log n)

    The harmonic series appears in the analysis of many algorithms,
    including quicksort's average case.

    Returns:
        BigThetaProof with integral bounds.
    """
    proof_steps = [
        "THEOREM: Hₙ = Σᵢ₌₁ⁿ 1/i ∈ Θ(ln n)",
        "",
        "We use integral bounds to estimate the harmonic series.",
        "",
        "INTEGRAL APPROXIMATION:",
        "Since 1/x is a decreasing function:",
        "",
        "∫₁ⁿ⁺¹ (1/x)dx ≤ Σᵢ₌₁ⁿ 1/i ≤ 1 + ∫₁ⁿ (1/x)dx",
        "",
        "Computing the integrals:",
        "Left: ln(n+1)",
        "Right: 1 + ln(n)",
        "",
        "Therefore: ln(n+1) ≤ Hₙ ≤ 1 + ln(n)",
        "",
        "UPPER BOUND (O(log n)):",
        "Hₙ ≤ 1 + ln(n) ≤ 2·ln(n)  for n ≥ 3 (since 1 ≤ ln(n) for n ≥ e)",
        "Choose c₂ = 2, n₀ = 3. ✓",
        "",
        "LOWER BOUND (Ω(log n)):",
        "Hₙ ≥ ln(n+1) ≥ ln(n)  for n ≥ 1",
        "And ln(n) ≥ (1/2)·ln(n)  trivially.",
        "Choose c₁ = 1/2, n₀ = 1. ✓",
        "",
        "More precisely, Euler showed: Hₙ = ln(n) + γ + O(1/n)",
        "where γ ≈ 0.5772 is the Euler-Mascheroni constant.",
        "",
        "CONCLUSION: Hₙ ∈ Θ(ln n) = Θ(log n). □",
        "",
        "APPLICATION:",
        "This explains why the expected number of comparisons in quicksort",
        "is O(n log n): each element is involved in O(log n) comparisons",
        "on average due to the harmonic structure of pivot selections.",
    ]

    # Verification
    verification = []
    for n in [1, 5, 10, 50, 100, 1000]:
        h_n = sum(1 / i for i in range(1, n + 1))  # Harmonic number
        log_n = math.log(n) if n > 0 else 1
        c1_g_n = 0.5 * log_n if log_n > 0 else 0.5
        c2_g_n = 2.0 * log_n if log_n > 0 else 2.0
        verification.append((n, h_n, c1_g_n, c2_g_n))

    return BigThetaProof(
        function_description="Hₙ = Σᵢ₌₁ⁿ 1/i (harmonic series)",
        big_theta_class="Θ(log n)",
        constant_c1=0.5,
        constant_c2=2.0,
        threshold_n0=3,
        proof_steps=proof_steps,
        verification_values=verification,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PROOF VERIFICATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def verify_big_o_numerically(
    f: Callable[[int], float],
    g: Callable[[int], float],
    c: float,
    n0: int,
    test_range: range = range(1, 1001),
) -> tuple[bool, list[int]]:
    """
    Numerically verify f(n) ≤ c·g(n) for n ≥ n₀.

    Args:
        f: Function f(n).
        g: Function g(n).
        c: Proposed constant c.
        n0: Proposed threshold n₀.
        test_range: Range of n values to test.

    Returns:
        Tuple of (passed, list of failing n values).
    """
    failures = []
    for n in test_range:
        if n >= n0:
            if f(n) > c * g(n):
                failures.append(n)

    return len(failures) == 0, failures


def verify_big_theta_numerically(
    f: Callable[[int], float],
    g: Callable[[int], float],
    c1: float,
    c2: float,
    n0: int,
    test_range: range = range(1, 1001),
) -> tuple[bool, list[int], list[int]]:
    """
    Numerically verify c₁·g(n) ≤ f(n) ≤ c₂·g(n) for n ≥ n₀.

    Args:
        f: Function f(n).
        g: Function g(n).
        c1: Lower bound constant.
        c2: Upper bound constant.
        n0: Threshold.
        test_range: Range of n values to test.

    Returns:
        Tuple of (passed, lower_failures, upper_failures).
    """
    lower_failures = []
    upper_failures = []

    for n in test_range:
        if n >= n0:
            f_n = f(n)
            g_n = g(n)

            if f_n < c1 * g_n:
                lower_failures.append(n)
            if f_n > c2 * g_n:
                upper_failures.append(n)

    passed = len(lower_failures) == 0 and len(upper_failures) == 0
    return passed, lower_failures, upper_failures


def print_proof(proof: BigOProof | BigThetaProof | RecurrenceProof) -> None:
    """Pretty-print a proof with formatting."""
    logger.info("=" * 70)

    if isinstance(proof, BigOProof):
        logger.info(f"BIG-O PROOF: {proof.function_description}")
        logger.info(f"Claim: f(n) ∈ {proof.big_o_class}")
        logger.info(f"Constants: c = {proof.constant_c}, n₀ = {proof.threshold_n0}")
    elif isinstance(proof, BigThetaProof):
        logger.info(f"BIG-Θ PROOF: {proof.function_description}")
        logger.info(f"Claim: f(n) ∈ {proof.big_theta_class}")
        logger.info(f"Constants: c₁ = {proof.constant_c1}, c₂ = {proof.constant_c2}, n₀ = {proof.threshold_n0}")
    elif isinstance(proof, RecurrenceProof):
        logger.info(f"RECURRENCE PROOF: {proof.recurrence}")
        logger.info(f"Solution: {proof.solution}")
        logger.info(f"Method: {proof.method}")

    logger.info("-" * 70)
    logger.info("PROOF:")
    for step in proof.proof_steps:
        logger.info(f"  {step}")

    logger.info("-" * 70)
    logger.info("VERIFICATION VALUES:")
    if isinstance(proof, BigThetaProof):
        logger.info(f"  {'n':>8} {'f(n)':>12} {'c₁·g(n)':>12} {'c₂·g(n)':>12}")
        for n, f_n, c1_g_n, c2_g_n in proof.verification_values:
            logger.info(f"  {n:>8} {f_n:>12.2f} {c1_g_n:>12.2f} {c2_g_n:>12.2f}")
    elif isinstance(proof, BigOProof):
        logger.info(f"  {'n':>8} {'f(n)':>12} {'c·g(n)':>12}")
        for n, f_n, c_g_n in proof.verification_values:
            logger.info(f"  {n:>8} {f_n:>12.2f} {c_g_n:>12.2f}")
    elif isinstance(proof, RecurrenceProof):
        logger.info(f"  {'n':>8} {'T(n)':>12}")
        for n, t_n in proof.verification_values:
            logger.info(f"  {n:>8} {t_n:>12}")

    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def run_all_solutions() -> None:
    """Run and display all complexity proof solutions."""
    logger.info("=" * 70)
    logger.info("COMPLEXITY PROOF SOLUTIONS - COMPLETE DEMONSTRATION")
    logger.info("=" * 70)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 1: Linear Search Proof")
    logger.info("─" * 70)
    proof = prove_linear_search()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 2: Bubble Sort Proof")
    logger.info("─" * 70)
    proof = prove_bubble_sort()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 3: Binary Search Proof")
    logger.info("─" * 70)
    proof = prove_binary_search()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 4: Merge Sort Recurrence")
    logger.info("─" * 70)
    proof = prove_merge_sort_recurrence()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 5: Prove n² ∉ O(n)")
    logger.info("─" * 70)
    proof = prove_not_in_big_o()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 6: Sum Complexity")
    logger.info("─" * 70)
    proof = prove_sum_complexity()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Bonus: log(n!) Complexity")
    logger.info("─" * 70)
    proof = prove_log_factorial()
    print_proof(proof)

    logger.info("\n" + "─" * 70)
    logger.info("Bonus: Harmonic Series")
    logger.info("─" * 70)
    proof = prove_harmonic_series()
    print_proof(proof)

    logger.info("\n" + "=" * 70)
    logger.info("ALL COMPLEXITY PROOF SOLUTIONS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Complexity Proof Exercise Solutions"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demonstration",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_all_solutions()
    else:
        logger.info("Run with --demo to see all solutions")
