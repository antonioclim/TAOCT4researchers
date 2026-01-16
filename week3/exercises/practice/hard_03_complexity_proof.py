#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Complexity Proofs (Hard)
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Understanding how to formally prove complexity bounds is essential for rigorous
algorithm analysis. This exercise develops your ability to construct mathematical
proofs of time and space complexity using the formal definitions of Big-O,
Big-Omega and Big-Theta notation.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Apply formal definitions of asymptotic notation
2. Construct proofs using limit analysis and the definition approach
3. Prove tight bounds using both upper and lower bound arguments
4. Identify common proof techniques and pitfalls

ESTIMATED TIME
──────────────
- Reading: 25 minutes
- Coding/Proving: 55 minutes
- Total: 80 minutes

DIFFICULTY: ⭐⭐⭐⭐⭐ (Hard)

BACKGROUND
──────────
Formal Definitions:
    
    O(g(n)): f(n) ∈ O(g(n)) iff ∃c > 0, n₀ > 0 such that
             ∀n ≥ n₀: f(n) ≤ c · g(n)
             
    Ω(g(n)): f(n) ∈ Ω(g(n)) iff ∃c > 0, n₀ > 0 such that
             ∀n ≥ n₀: f(n) ≥ c · g(n)
             
    Θ(g(n)): f(n) ∈ Θ(g(n)) iff f(n) ∈ O(g(n)) AND f(n) ∈ Ω(g(n))
             Equivalently: ∃c₁, c₂ > 0, n₀ > 0 such that
             ∀n ≥ n₀: c₁ · g(n) ≤ f(n) ≤ c₂ · g(n)

Limit Method:
    If lim(n→∞) f(n)/g(n) = L where 0 < L < ∞, then f(n) ∈ Θ(g(n))
    If lim(n→∞) f(n)/g(n) = 0, then f(n) ∈ O(g(n)) but f(n) ∉ Ω(g(n))
    If lim(n→∞) f(n)/g(n) = ∞, then f(n) ∈ Ω(g(n)) but f(n) ∉ O(g(n))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROOF STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BigOProof:
    """Structure for a Big-O proof."""
    function_name: str
    f_n: str  # Function being analysed (as string)
    g_n: str  # Upper bound function
    c: float  # Constant multiplier
    n0: int   # Threshold value
    proof_text: str  # The actual proof
    
    def verify_numerically(
        self, 
        f: Callable[[int], float], 
        g: Callable[[int], float],
        test_values: list[int] | None = None
    ) -> bool:
        """
        Numerically verify that f(n) ≤ c·g(n) for n ≥ n₀.
        
        Args:
            f: The function f(n).
            g: The function g(n).
            test_values: Values to test (default: [n₀, n₀+10, n₀+100, ...])
            
        Returns:
            True if verification passes for all test values.
        """
        if test_values is None:
            test_values = [
                self.n0, 
                self.n0 + 10, 
                self.n0 + 100, 
                self.n0 * 10,
                self.n0 * 100
            ]
        
        for n in test_values:
            if n >= self.n0:
                f_val = f(n)
                bound = self.c * g(n)
                if f_val > bound:
                    logger.warning(
                        f"Verification failed: f({n})={f_val} > "
                        f"c·g({n})={bound}"
                    )
                    return False
        
        return True


@dataclass
class BigThetaProof:
    """Structure for a Big-Θ proof (tight bound)."""
    function_name: str
    f_n: str
    g_n: str
    c1: float  # Lower constant
    c2: float  # Upper constant
    n0: int
    proof_text: str
    
    def verify_numerically(
        self,
        f: Callable[[int], float],
        g: Callable[[int], float],
        test_values: list[int] | None = None
    ) -> bool:
        """Verify c₁·g(n) ≤ f(n) ≤ c₂·g(n) for n ≥ n₀."""
        if test_values is None:
            test_values = [self.n0, self.n0 * 2, self.n0 * 10, self.n0 * 100]
        
        for n in test_values:
            if n >= self.n0:
                f_val = f(n)
                lower = self.c1 * g(n)
                upper = self.c2 * g(n)
                
                if f_val < lower or f_val > upper:
                    logger.warning(
                        f"Verification failed at n={n}: "
                        f"{lower} ≤ {f_val} ≤ {upper} is False"
                    )
                    return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXAMPLE PROOFS
# ═══════════════════════════════════════════════════════════════════════════════

def example_proof_polynomial() -> BigOProof:
    """
    Example: Prove that 3n² + 5n + 7 ∈ O(n²).
    
    This demonstrates the standard technique for polynomial bounds.
    """
    proof_text = """
    CLAIM: 3n² + 5n + 7 ∈ O(n²)
    
    PROOF:
    We need to find c > 0 and n₀ > 0 such that for all n ≥ n₀:
        3n² + 5n + 7 ≤ c · n²
    
    For n ≥ 1:
        5n ≤ 5n²     (since n ≤ n² when n ≥ 1)
        7 ≤ 7n²     (since 1 ≤ n² when n ≥ 1)
    
    Therefore:
        3n² + 5n + 7 ≤ 3n² + 5n² + 7n² = 15n²
    
    Choose c = 15, n₀ = 1.
    
    For all n ≥ 1: 3n² + 5n + 7 ≤ 15n²
    
    Thus, 3n² + 5n + 7 ∈ O(n²). ∎
    """
    
    return BigOProof(
        function_name="polynomial_example",
        f_n="3n² + 5n + 7",
        g_n="n²",
        c=15,
        n0=1,
        proof_text=proof_text
    )


def example_proof_logarithm() -> BigOProof:
    """
    Example: Prove that log₂(n) ∈ O(n).
    
    This demonstrates handling logarithms.
    """
    proof_text = """
    CLAIM: log₂(n) ∈ O(n)
    
    PROOF:
    We use the limit definition as an alternative approach:
    
    lim(n→∞) log₂(n) / n
    
    Using L'Hôpital's rule (differentiating numerator and denominator):
    = lim(n→∞) (1/(n·ln(2))) / 1
    = lim(n→∞) 1/(n·ln(2))
    = 0
    
    Since the limit is 0, log₂(n) = o(n) (little-o), which implies
    log₂(n) ∈ O(n).
    
    For explicit constants, note that log₂(n) ≤ n for all n ≥ 1.
    Choose c = 1, n₀ = 1.
    
    Thus, log₂(n) ∈ O(n). ∎
    """
    
    return BigOProof(
        function_name="logarithm_example",
        f_n="log₂(n)",
        g_n="n",
        c=1,
        n0=1,
        proof_text=proof_text
    )


def example_proof_tight_bound() -> BigThetaProof:
    """
    Example: Prove that n² - 3n + 2 ∈ Θ(n²).
    
    This demonstrates proving tight bounds.
    """
    proof_text = """
    CLAIM: n² - 3n + 2 ∈ Θ(n²)
    
    PROOF:
    We need to prove both O(n²) and Ω(n²).
    
    UPPER BOUND (O(n²)):
    For n ≥ 1:
        n² - 3n + 2 ≤ n² + 2 ≤ n² + 2n² = 3n²
    Choose c₂ = 3, verified for n ≥ 1.
    
    LOWER BOUND (Ω(n²)):
    For n ≥ 6:
        3n ≤ n²/2  (since n ≥ 6)
        -3n ≥ -n²/2
        n² - 3n + 2 ≥ n² - n²/2 + 2 = n²/2 + 2 ≥ n²/2
    Choose c₁ = 1/2, verified for n ≥ 6.
    
    CONCLUSION:
    With c₁ = 1/2, c₂ = 3, n₀ = 6:
        ∀n ≥ 6: (1/2)n² ≤ n² - 3n + 2 ≤ 3n²
    
    Thus, n² - 3n + 2 ∈ Θ(n²). ∎
    """
    
    return BigThetaProof(
        function_name="tight_bound_example",
        f_n="n² - 3n + 2",
        g_n="n²",
        c1=0.5,
        c2=3,
        n0=6,
        proof_text=proof_text
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: YOUR PROOF TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def prove_linear_search() -> BigThetaProof:
    """
    EXERCISE 1: Prove the complexity of linear search.
    
    Linear search examines elements one by one until finding the target
    or reaching the end. The number of comparisons in the worst case
    is exactly n.
    
    TASK: Prove that T(n) = n ∈ Θ(n)
    
    This is straightforward but demonstrates the proof structure.
    Find appropriate c₁, c₂ and n₀.
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this proof
    proof_text = """
    CLAIM: T(n) = n ∈ Θ(n)
    
    PROOF:
    [YOUR PROOF HERE]
    
    [Show upper bound: n ≤ c₂·n]
    [Show lower bound: n ≥ c₁·n]
    [Specify c₁, c₂, n₀]
    
    ∎
    """
    
    raise NotImplementedError("Complete the proof for linear search")


def prove_bubble_sort() -> BigThetaProof:
    """
    EXERCISE 2: Prove the worst-case complexity of bubble sort.
    
    In the worst case (reverse-sorted array), bubble sort performs:
        T(n) = (n-1) + (n-2) + ... + 1 = n(n-1)/2 comparisons
    
    TASK: Prove that n(n-1)/2 ∈ Θ(n²)
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this proof
    proof_text = """
    CLAIM: T(n) = n(n-1)/2 ∈ Θ(n²)
    
    PROOF:
    First, expand: n(n-1)/2 = (n² - n)/2 = n²/2 - n/2
    
    UPPER BOUND:
    [YOUR PROOF HERE]
    
    LOWER BOUND:
    [YOUR PROOF HERE]
    
    CONCLUSION:
    [State c₁, c₂, n₀ and conclude]
    
    ∎
    """
    
    raise NotImplementedError("Complete the proof for bubble sort")


def prove_binary_search() -> BigOProof:
    """
    EXERCISE 3: Prove binary search complexity.
    
    Binary search halves the search space on each iteration.
    After k iterations, the remaining size is n/2^k.
    The algorithm terminates when n/2^k ≤ 1, i.e., k ≥ log₂(n).
    
    TASK: Prove that T(n) = ⌊log₂(n)⌋ + 1 ∈ O(log n)
    
    Note: Use ⌊log₂(n)⌋ ≤ log₂(n) to simplify.
    
    Returns:
        A BigOProof object with your proof.
    """
    # TODO: Complete this proof
    proof_text = """
    CLAIM: T(n) = ⌊log₂(n)⌋ + 1 ∈ O(log n)
    
    PROOF:
    [YOUR PROOF HERE]
    
    ∎
    """
    
    raise NotImplementedError("Complete the proof for binary search")


def prove_merge_sort_recurrence() -> BigThetaProof:
    """
    EXERCISE 4: Prove merge sort complexity via recurrence.
    
    Merge sort satisfies the recurrence:
        T(n) = 2T(n/2) + n
        T(1) = 1
    
    TASK: Use the Master Theorem to prove T(n) ∈ Θ(n log n)
    
    Master Theorem: For T(n) = aT(n/b) + f(n)
        - If f(n) = O(n^(log_b(a) - ε)), then T(n) = Θ(n^(log_b(a)))
        - If f(n) = Θ(n^(log_b(a))), then T(n) = Θ(n^(log_b(a)) log n)
        - If f(n) = Ω(n^(log_b(a) + ε)), then T(n) = Θ(f(n))
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this proof
    proof_text = """
    CLAIM: T(n) = 2T(n/2) + n ∈ Θ(n log n)
    
    PROOF using Master Theorem:
    Identify: a = ?, b = ?, f(n) = ?
    
    Compute: log_b(a) = ?
    
    Compare f(n) with n^(log_b(a)):
    [YOUR ANALYSIS HERE]
    
    Apply appropriate case:
    [YOUR CONCLUSION HERE]
    
    ∎
    """
    
    raise NotImplementedError("Complete the proof for merge sort")


def prove_not_in_big_o() -> str:
    """
    EXERCISE 5: Prove that n² ∉ O(n).
    
    Sometimes you need to prove a function is NOT in a complexity class.
    This requires showing no constants c, n₀ can satisfy the definition.
    
    TASK: Prove that n² ∉ O(n) using proof by contradiction.
    
    Returns:
        A string containing your complete proof.
    """
    # TODO: Complete this proof
    proof = """
    CLAIM: n² ∉ O(n)
    
    PROOF BY CONTRADICTION:
    Assume n² ∈ O(n).
    
    Then ∃c > 0, n₀ > 0 such that ∀n ≥ n₀: n² ≤ c·n
    
    [CONTINUE YOUR PROOF - derive a contradiction]
    
    [Show that for large enough n, n² > c·n regardless of c]
    
    This contradicts our assumption.
    
    Therefore, n² ∉ O(n). ∎
    """
    
    raise NotImplementedError("Complete the proof that n² ∉ O(n)")


def prove_sum_complexity() -> BigThetaProof:
    """
    EXERCISE 6: Prove the complexity of summing an arithmetic series.
    
    Consider the sum: S(n) = 1 + 2 + 3 + ... + n = n(n+1)/2
    
    TASK: Prove that S(n) ∈ Θ(n²)
    
    This is similar to bubble sort but requires careful handling of +1.
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this proof
    proof_text = """
    CLAIM: S(n) = n(n+1)/2 ∈ Θ(n²)
    
    PROOF:
    [YOUR PROOF HERE]
    
    ∎
    """
    
    raise NotImplementedError("Complete the proof for arithmetic sum")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ADVANCED PROOF TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def prove_log_factorial() -> BigThetaProof:
    """
    BONUS 1: Prove that log(n!) ∈ Θ(n log n).
    
    This is important for understanding comparison-based sorting
    lower bounds.
    
    Hint: Use Stirling's approximation: n! ≈ √(2πn)(n/e)^n
    Therefore: log(n!) ≈ n log n - n log e + (1/2) log(2πn)
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this advanced proof
    raise NotImplementedError("Complete the proof for log(n!)")


def prove_harmonic_series() -> BigThetaProof:
    """
    BONUS 2: Prove that H(n) = 1 + 1/2 + 1/3 + ... + 1/n ∈ Θ(log n).
    
    The harmonic series appears in the analysis of many algorithms,
    including quicksort's expected case.
    
    Hint: Use integral bounds:
        ∫₁ⁿ (1/x) dx ≤ H(n) ≤ 1 + ∫₁ⁿ (1/x) dx
        ln(n) ≤ H(n) ≤ 1 + ln(n)
    
    Returns:
        A BigThetaProof object with your proof.
    """
    # TODO: Complete this advanced proof
    raise NotImplementedError("Complete the proof for harmonic series")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VERIFICATION AND DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def verify_example_proofs() -> bool:
    """Verify that example proofs are correct."""
    # Polynomial example
    poly_proof = example_proof_polynomial()
    f_poly = lambda n: 3*n**2 + 5*n + 7
    g_poly = lambda n: n**2
    assert poly_proof.verify_numerically(f_poly, g_poly), "Polynomial proof failed"
    
    # Logarithm example
    log_proof = example_proof_logarithm()
    f_log = lambda n: math.log2(n) if n > 0 else 0
    g_log = lambda n: n
    assert log_proof.verify_numerically(f_log, g_log, [2, 10, 100, 1000]), \
        "Logarithm proof failed"
    
    # Tight bound example
    tight_proof = example_proof_tight_bound()
    f_tight = lambda n: n**2 - 3*n + 2
    g_tight = lambda n: n**2
    assert tight_proof.verify_numerically(f_tight, g_tight), "Tight bound proof failed"
    
    logger.info("All example proofs verified successfully!")
    return True


def demo() -> None:
    """Demonstrate complexity proof techniques."""
    logger.info("=" * 70)
    logger.info("COMPLEXITY PROOFS DEMONSTRATION")
    logger.info("=" * 70)
    
    # Demo 1: Polynomial proof
    logger.info("\nDemo 1: Polynomial Big-O Proof")
    poly_proof = example_proof_polynomial()
    logger.info(poly_proof.proof_text)
    
    # Numerical verification
    f = lambda n: 3*n**2 + 5*n + 7
    g = lambda n: n**2
    
    logger.info("Numerical verification:")
    for n in [1, 10, 100, 1000]:
        f_val = f(n)
        bound = poly_proof.c * g(n)
        logger.info(f"  n={n}: f(n)={f_val}, {poly_proof.c}·g(n)={bound}, OK={f_val <= bound}")
    
    # Demo 2: Tight bound proof
    logger.info("\nDemo 2: Tight Bound (Θ) Proof")
    tight_proof = example_proof_tight_bound()
    logger.info(tight_proof.proof_text)
    
    f = lambda n: n**2 - 3*n + 2
    g = lambda n: n**2
    
    logger.info("Numerical verification:")
    for n in [6, 10, 100, 1000]:
        f_val = f(n)
        lower = tight_proof.c1 * g(n)
        upper = tight_proof.c2 * g(n)
        logger.info(
            f"  n={n}: {lower:.0f} ≤ {f_val} ≤ {upper:.0f}, "
            f"OK={lower <= f_val <= upper}"
        )
    
    # Demo 3: Limit method
    logger.info("\nDemo 3: Using the Limit Method")
    logger.info("For f(n) = n² + 100n vs g(n) = n²:")
    logger.info("lim(n→∞) (n² + 100n)/n² = lim(n→∞) 1 + 100/n = 1")
    logger.info("Since 0 < 1 < ∞, we have n² + 100n ∈ Θ(n²)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Complete the exercises to master complexity proofs!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complexity Proofs Practice Exercise"
    )
    parser.add_argument("--verify", action="store_true", help="Verify example proofs")
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    args = parser.parse_args()
    
    if args.verify:
        verify_example_proofs()
    elif args.demo:
        demo()
    else:
        logger.info("Use --verify to test or --demo to see demonstrations")


if __name__ == "__main__":
    main()
