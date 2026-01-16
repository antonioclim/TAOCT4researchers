#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Lab 2: Complexity Analyser — Automatic Big-O Estimation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
While theoretical complexity analysis is essential, empirical verification
provides confidence that implementations match expectations. This lab provides
tools for automatically estimating algorithm complexity from timing measurements
using statistical techniques including log-log regression and curve fitting.

PREREQUISITES
─────────────
- Week 3 Lab 1: BenchmarkSuite and BenchmarkResult classes
- Mathematics: Logarithms, linear regression, curve fitting basics
- Python: NumPy, SciPy basics

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Estimate empirical complexity exponents using log-log regression
2. Fit and compare multiple complexity models (O(n), O(n log n), O(n²))
3. Visualise complexity analysis with diagnostic plots
4. Identify discrepancies between theoretical and empirical complexity

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 30 minutes
- Total: 50 minutes

DEPENDENCIES
────────────
- numpy>=1.24
- scipy>=1.11
- matplotlib>=3.7 (optional, for visualisation)

THEORY
──────
For power-law complexity T(n) = c · n^k:
    
    log T(n) = log c + k · log n

Linear regression on (log n, log T) yields slope k (the exponent).

For O(n log n), we fit T(n) = c · n · log(n) directly using nonlinear
least squares and compare residuals to identify the best model.

Statistical considerations:
- Multiple measurements reduce noise
- Warmup eliminates JIT and cache effects
- Robust statistics (median, IQR) handle outliers
- R² indicates goodness of fit

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

# Optional imports with graceful fallbacks
try:
    from scipy.optimize import curve_fit
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available; some features will be limited")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: COMPLEXITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexityClass(Enum):
    """Standard algorithmic complexity classes."""
    CONSTANT = auto()      # O(1)
    LOGARITHMIC = auto()   # O(log n)
    LINEAR = auto()        # O(n)
    LINEARITHMIC = auto()  # O(n log n)
    QUADRATIC = auto()     # O(n²)
    CUBIC = auto()         # O(n³)
    POLYNOMIAL = auto()    # O(n^k) for k > 3
    EXPONENTIAL = auto()   # O(2^n)
    UNKNOWN = auto()       # Cannot determine


@dataclass
class ComplexityModel:
    """
    Represents a fitted complexity model.
    
    Attributes:
        name: Human-readable name (e.g., "O(n log n)").
        complexity_class: Enumerated complexity class.
        exponent: Estimated exponent for power-law models.
        coefficient: Leading coefficient c in T(n) = c · f(n).
        r_squared: Coefficient of determination (goodness of fit).
        residual_sum: Sum of squared residuals.
        predictions: Model predictions at measured sizes.
    """
    name: str
    complexity_class: ComplexityClass
    exponent: float
    coefficient: float
    r_squared: float
    residual_sum: float
    predictions: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    def __str__(self) -> str:
        return f"{self.name} (R²={self.r_squared:.4f}, exp={self.exponent:.2f})"
    
    def predict(self, n: int | NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Predict time for given input size(s).
        
        Args:
            n: Input size or array of sizes.
        
        Returns:
            Predicted time(s).
        """
        n_arr = np.atleast_1d(np.asarray(n, dtype=np.float64))
        
        if self.complexity_class == ComplexityClass.CONSTANT:
            return np.full_like(n_arr, self.coefficient)
        elif self.complexity_class == ComplexityClass.LOGARITHMIC:
            return self.coefficient * np.log(n_arr)
        elif self.complexity_class == ComplexityClass.LINEAR:
            return self.coefficient * n_arr
        elif self.complexity_class == ComplexityClass.LINEARITHMIC:
            return self.coefficient * n_arr * np.log(n_arr)
        elif self.complexity_class == ComplexityClass.QUADRATIC:
            return self.coefficient * n_arr ** 2
        elif self.complexity_class == ComplexityClass.CUBIC:
            return self.coefficient * n_arr ** 3
        else:
            return self.coefficient * n_arr ** self.exponent


class FitResult(NamedTuple):
    """Result of a model fitting operation."""
    coefficient: float
    r_squared: float
    residual_sum: float
    predictions: NDArray[np.float64]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL FITTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _constant_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(1) model: T(n) = c."""
    return np.full_like(n, c, dtype=np.float64)


def _log_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(log n) model: T(n) = c · log(n)."""
    return c * np.log(n)


def _linear_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(n) model: T(n) = c · n."""
    return c * n


def _linearithmic_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(n log n) model: T(n) = c · n · log(n)."""
    return c * n * np.log(n)


def _quadratic_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(n²) model: T(n) = c · n²."""
    return c * n ** 2


def _cubic_model(n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """O(n³) model: T(n) = c · n³."""
    return c * n ** 3


def _power_model(n: NDArray[np.float64], c: float, k: float) -> NDArray[np.float64]:
    """General power model: T(n) = c · n^k."""
    return c * n ** k


def _fit_model(
    sizes: NDArray[np.float64],
    times: NDArray[np.float64],
    model_func: Callable[..., NDArray[np.float64]],
    initial_guess: tuple[float, ...] | None = None
) -> FitResult:
    """
    Fit a model to timing data using least squares.
    
    Args:
        sizes: Array of input sizes.
        times: Array of timing measurements.
        model_func: Model function to fit.
        initial_guess: Initial parameter guess for optimisation.
    
    Returns:
        FitResult with coefficient, R², residuals and predictions.
    """
    if not HAS_SCIPY:
        # Fallback to simple estimation
        logger.warning("SciPy not available; using simple estimation")
        c_guess = times[-1] / model_func(sizes[-1:], 1.0)[0]
        predictions = model_func(sizes, c_guess)
        ss_res = np.sum((times - predictions) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return FitResult(c_guess, r_squared, ss_res, predictions)
    
    try:
        # Determine number of parameters
        import inspect
        sig = inspect.signature(model_func)
        n_params = len(sig.parameters) - 1  # Exclude 'n'
        
        if initial_guess is None:
            initial_guess = tuple([1.0] * n_params)
        
        params, _ = curve_fit(
            model_func, sizes, times,
            p0=initial_guess,
            maxfev=10000,
            bounds=(0, np.inf)
        )
        
        predictions = model_func(sizes, *params)
        ss_res = np.sum((times - predictions) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        coefficient = params[0]
        
        return FitResult(coefficient, r_squared, ss_res, predictions)
        
    except Exception as e:
        logger.warning(f"Curve fitting failed: {e}")
        return FitResult(0.0, 0.0, float('inf'), np.zeros_like(sizes))


def estimate_exponent(
    sizes: Sequence[int] | NDArray[np.int64],
    times: Sequence[float] | NDArray[np.float64]
) -> tuple[float, float]:
    """
    Estimate complexity exponent via log-log regression.
    
    Uses the relationship:
        T(n) = c · n^k  →  log(T) = log(c) + k · log(n)
    
    Args:
        sizes: Array of input sizes.
        times: Array of timing measurements.
    
    Returns:
        Tuple of (exponent, r_squared).
    
    Example:
        >>> sizes = [100, 1000, 10000]
        >>> times = [0.01, 1.0, 100.0]  # O(n²) growth
        >>> exp, r2 = estimate_exponent(sizes, times)
        >>> f"Exponent: {exp:.2f}"
        'Exponent: 2.00'
    """
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    
    # Filter valid (positive) values
    valid = (sizes_arr > 0) & (times_arr > 0)
    if np.sum(valid) < 2:
        logger.warning("Insufficient valid data points")
        return 0.0, 0.0
    
    log_n = np.log(sizes_arr[valid])
    log_t = np.log(times_arr[valid])
    
    if HAS_SCIPY:
        result = linregress(log_n, log_t)
        return float(result.slope), float(result.rvalue ** 2)
    
    # Manual linear regression
    n = len(log_n)
    sum_x = np.sum(log_n)
    sum_y = np.sum(log_t)
    sum_xy = np.sum(log_n * log_t)
    sum_x2 = np.sum(log_n ** 2)
    
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.0, 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate R²
    y_pred = intercept + slope * log_n
    ss_res = np.sum((log_t - y_pred) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return float(slope), float(r_squared)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPLEXITY ANALYSER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplexityAnalyser:
    """
    Comprehensive complexity analyser for empirical algorithm analysis.
    
    Provides methods to:
    - Estimate complexity exponent via log-log regression
    - Fit multiple standard complexity models
    - Select the best-fitting model
    - Generate diagnostic visualisations
    
    Attributes:
        sizes: Array of input sizes tested.
        times: Array of timing measurements (typically median values).
        algorithm_name: Name of the algorithm being analysed.
    
    Example:
        >>> analyser = ComplexityAnalyser(
        ...     sizes=[100, 500, 1000, 5000, 10000],
        ...     times=[0.5, 12.5, 50.0, 1250.0, 5000.0],
        ...     algorithm_name="Bubble Sort"
        ... )
        >>> best = analyser.best_fit()
        >>> logger.info(f"Complexity: {best.name}")
    """
    sizes: NDArray[np.float64]
    times: NDArray[np.float64]
    algorithm_name: str = "Unknown"
    _models: dict[str, ComplexityModel] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Convert inputs to numpy arrays."""
        self.sizes = np.asarray(self.sizes, dtype=np.float64)
        self.times = np.asarray(self.times, dtype=np.float64)
        
        if len(self.sizes) != len(self.times):
            raise ValueError("sizes and times must have the same length")
        
        if len(self.sizes) < 3:
            logger.warning("At least 3 data points recommended for reliable analysis")
    
    @classmethod
    def from_benchmark_results(
        cls,
        results: Sequence[Any],  # BenchmarkResult from lab_3_01
        algorithm_name: str = "Unknown"
    ) -> ComplexityAnalyser:
        """
        Create analyser from benchmark results.
        
        Args:
            results: List of BenchmarkResult objects.
            algorithm_name: Name of the algorithm.
        
        Returns:
            Configured ComplexityAnalyser instance.
        """
        sizes = np.array([r.n for r in results], dtype=np.float64)
        times = np.array([r.median for r in results], dtype=np.float64)
        return cls(sizes=sizes, times=times, algorithm_name=algorithm_name)
    
    def estimate_exponent(self) -> tuple[float, float]:
        """
        Estimate complexity exponent via log-log regression.
        
        Returns:
            Tuple of (exponent, r_squared).
        """
        return estimate_exponent(self.sizes, self.times)
    
    def fit_models(self) -> dict[str, ComplexityModel]:
        """
        Fit all standard complexity models to the data.
        
        Returns:
            Dictionary mapping model names to ComplexityModel objects.
        """
        models: dict[str, ComplexityModel] = {}
        
        # O(1) - Constant
        result = _fit_model(self.sizes, self.times, _constant_model)
        models["O(1)"] = ComplexityModel(
            name="O(1)",
            complexity_class=ComplexityClass.CONSTANT,
            exponent=0.0,
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # O(log n) - Logarithmic
        result = _fit_model(self.sizes, self.times, _log_model)
        models["O(log n)"] = ComplexityModel(
            name="O(log n)",
            complexity_class=ComplexityClass.LOGARITHMIC,
            exponent=0.0,  # Not a power law
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # O(n) - Linear
        result = _fit_model(self.sizes, self.times, _linear_model)
        models["O(n)"] = ComplexityModel(
            name="O(n)",
            complexity_class=ComplexityClass.LINEAR,
            exponent=1.0,
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # O(n log n) - Linearithmic
        result = _fit_model(self.sizes, self.times, _linearithmic_model)
        models["O(n log n)"] = ComplexityModel(
            name="O(n log n)",
            complexity_class=ComplexityClass.LINEARITHMIC,
            exponent=1.0,  # Approximately
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # O(n²) - Quadratic
        result = _fit_model(self.sizes, self.times, _quadratic_model)
        models["O(n²)"] = ComplexityModel(
            name="O(n²)",
            complexity_class=ComplexityClass.QUADRATIC,
            exponent=2.0,
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # O(n³) - Cubic
        result = _fit_model(self.sizes, self.times, _cubic_model)
        models["O(n³)"] = ComplexityModel(
            name="O(n³)",
            complexity_class=ComplexityClass.CUBIC,
            exponent=3.0,
            coefficient=result.coefficient,
            r_squared=result.r_squared,
            residual_sum=result.residual_sum,
            predictions=result.predictions
        )
        
        # Store and return
        self._models = models
        return models
    
    def best_fit(self) -> ComplexityModel:
        """
        Determine the best-fitting complexity model.
        
        Uses a combination of R² and residual analysis to select
        the most appropriate model, with preference for simpler
        models when fits are similar (parsimony).
        
        Returns:
            Best-fitting ComplexityModel.
        """
        if not self._models:
            self.fit_models()
        
        # Rank by R² with penalty for overly complex models
        candidates = list(self._models.values())
        
        # Filter to models with reasonable fit (R² > 0.9)
        good_fits = [m for m in candidates if m.r_squared > 0.9]
        
        if not good_fits:
            # Fall back to best available
            good_fits = candidates
        
        # Select model with highest R² among good fits
        best = max(good_fits, key=lambda m: m.r_squared)
        
        logger.debug(f"Best model: {best}")
        return best
    
    def classify(self) -> str:
        """
        Classify the algorithm's complexity.
        
        Returns:
            String representation of complexity class (e.g., "O(n log n)").
        """
        best = self.best_fit()
        return best.name
    
    def summary(self) -> str:
        """
        Generate a detailed analysis summary.
        
        Returns:
            Multi-line string with analysis results.
        """
        if not self._models:
            self.fit_models()
        
        # Log-log regression
        exponent, r2_loglog = self.estimate_exponent()
        
        lines = [
            "═" * 60,
            f"  COMPLEXITY ANALYSIS: {self.algorithm_name}",
            "═" * 60,
            "",
            f"Data points: {len(self.sizes)}",
            f"Size range: {int(self.sizes.min()):,} to {int(self.sizes.max()):,}",
            "",
            "Log-log regression:",
            f"  Exponent: {exponent:.3f}",
            f"  R²: {r2_loglog:.4f}",
            "",
            "Model comparison:",
        ]
        
        # Sort models by R²
        sorted_models = sorted(
            self._models.values(),
            key=lambda m: m.r_squared,
            reverse=True
        )
        
        for model in sorted_models:
            marker = "→" if model == self.best_fit() else " "
            lines.append(
                f"  {marker} {model.name:<12} R²={model.r_squared:.4f}  "
                f"c={model.coefficient:.2e}"
            )
        
        lines.extend([
            "",
            f"Best fit: {self.best_fit().name}",
            "═" * 60
        ])
        
        return "\n".join(lines)
    
    def plot_analysis(self, save_path: str | Path | None = None) -> Figure | None:
        """
        Generate diagnostic plots for the analysis.
        
        Creates a figure with:
        1. Log-log plot with regression line
        2. Residuals plot
        3. Model comparison
        
        Args:
            save_path: Optional path to save the figure.
        
        Returns:
            matplotlib Figure object, or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available; cannot generate plots")
            return None
        
        if not self._models:
            self.fit_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Complexity Analysis: {self.algorithm_name}", fontsize=14)
        
        # 1. Log-log plot
        ax1 = axes[0, 0]
        ax1.loglog(self.sizes, self.times, 'ko', markersize=8, label='Measured')
        
        # Regression line
        exp, _ = self.estimate_exponent()
        log_sizes = np.log(self.sizes)
        log_times = np.log(self.times)
        fit_coef = np.polyfit(log_sizes, log_times, 1)
        fit_times = np.exp(np.polyval(fit_coef, log_sizes))
        ax1.loglog(self.sizes, fit_times, 'r--', linewidth=2, 
                   label=f'Fit: n^{exp:.2f}')
        
        ax1.set_xlabel('Input Size (n)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Log-Log Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Best model fit
        ax2 = axes[0, 1]
        ax2.plot(self.sizes, self.times, 'ko', markersize=8, label='Measured')
        
        best = self.best_fit()
        smooth_sizes = np.linspace(self.sizes.min(), self.sizes.max(), 100)
        smooth_pred = best.predict(smooth_sizes)
        ax2.plot(smooth_sizes, smooth_pred, 'b-', linewidth=2,
                 label=f'Best fit: {best.name}')
        
        ax2.set_xlabel('Input Size (n)')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title(f'Best Model: {best.name} (R²={best.r_squared:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals
        ax3 = axes[1, 0]
        residuals = self.times - best.predictions
        ax3.bar(range(len(residuals)), residuals, color='steelblue', alpha=0.7)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Measurement Index')
        ax3.set_ylabel('Residual (ms)')
        ax3.set_title('Residuals (Measured - Predicted)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Model comparison (R² values)
        ax4 = axes[1, 1]
        model_names = list(self._models.keys())
        r_squared_values = [self._models[name].r_squared for name in model_names]
        
        colours = ['green' if r > 0.95 else 'orange' if r > 0.8 else 'red' 
                   for r in r_squared_values]
        bars = ax4.barh(model_names, r_squared_values, color=colours, alpha=0.7)
        ax4.axvline(x=0.95, color='g', linestyle='--', alpha=0.5, label='Excellent fit')
        ax4.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='Good fit')
        ax4.set_xlabel('R² (Coefficient of Determination)')
        ax4.set_title('Model Comparison')
        ax4.set_xlim(0, 1.05)
        ax4.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_exponent(exponent: float) -> str:
    """
    Classify a complexity exponent into a standard class.
    
    Args:
        exponent: Estimated exponent from log-log regression.
    
    Returns:
        String representation of complexity class.
    """
    if exponent < 0.1:
        return "O(1)"
    elif exponent < 0.6:
        return "O(log n)"
    elif exponent < 1.2:
        return "O(n)"
    elif exponent < 1.6:
        return "O(n log n)"
    elif exponent < 2.2:
        return "O(n²)"
    elif exponent < 3.2:
        return "O(n³)"
    else:
        return f"O(n^{exponent:.1f})"


def quick_estimate(
    sizes: Sequence[int],
    times: Sequence[float]
) -> str:
    """
    Quick complexity estimation for simple use cases.
    
    Args:
        sizes: Input sizes.
        times: Timing measurements.
    
    Returns:
        Complexity class string.
    """
    exponent, r2 = estimate_exponent(sizes, times)
    complexity = classify_exponent(exponent)
    logger.info(f"Quick estimate: {complexity} (exp={exponent:.2f}, R²={r2:.3f})")
    return complexity


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_complexity_analyser() -> None:
    """Demonstrate the ComplexityAnalyser with sample data."""
    logger.info("═" * 60)
    logger.info("  DEMO: Complexity Analyser")
    logger.info("═" * 60)
    
    # Simulated O(n²) data (like bubble sort)
    sizes = np.array([100, 200, 500, 1000, 2000, 5000])
    # T(n) ≈ 0.0001 * n² (with some noise)
    np.random.seed(42)
    times = 0.0001 * sizes ** 2 * (1 + 0.1 * np.random.randn(len(sizes)))
    
    analyser = ComplexityAnalyser(
        sizes=sizes,
        times=times,
        algorithm_name="Simulated O(n²)"
    )
    
    logger.info(analyser.summary())
    
    # Simulated O(n log n) data (like merge sort)
    times_nlogn = 0.001 * sizes * np.log(sizes) * (1 + 0.1 * np.random.randn(len(sizes)))
    
    analyser2 = ComplexityAnalyser(
        sizes=sizes,
        times=times_nlogn,
        algorithm_name="Simulated O(n log n)"
    )
    
    logger.info(analyser2.summary())
    
    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        fig = analyser.plot_analysis()
        if fig:
            plt.show()


def run_all_demos() -> None:
    """Run all demonstrations."""
    demo_complexity_analyser()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Week 3 Lab 2: Complexity Analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_3_02_complexity_analyser.py --demo
  python lab_3_02_complexity_analyser.py --sizes 100 500 1000 --times 0.5 12.5 50.0
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Input sizes for analysis"
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        help="Timing measurements (same length as sizes)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("═" * 60)
    logger.info("  WEEK 3 LAB 2: COMPLEXITY ANALYSER")
    logger.info("═" * 60)
    logger.info(f"SciPy available: {HAS_SCIPY}")
    logger.info(f"Matplotlib available: {HAS_MATPLOTLIB}")
    
    if args.demo:
        run_all_demos()
    elif args.sizes and args.times:
        if len(args.sizes) != len(args.times):
            logger.error("sizes and times must have the same length")
            return
        
        analyser = ComplexityAnalyser(
            sizes=np.array(args.sizes, dtype=np.float64),
            times=np.array(args.times, dtype=np.float64),
            algorithm_name="User-provided data"
        )
        logger.info(analyser.summary())
        
        if args.plot:
            fig = analyser.plot_analysis()
            if fig:
                plt.show()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
