"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7: Reproducibility and Capstone - Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package contains the laboratory materials for Week 7, covering
reproducibility, testing and project scaffolding.

Modules:
    lab_7_01_reproducibility: Seed management and data manifests
    lab_7_02_testing_cicd: Testing best practices and CI/CD
    lab_7_03_project_scaffolder: Automatic project structure generation

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
"""

from .lab_7_01_reproducibility import (
    DataManifest,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ReproducibilityConfig,
    compute_file_hash,
    generate_readme,
    retry,
    set_all_seeds,
    timed,
    verify_file_hash,
)
from .lab_7_02_testing_cicd import (
    Calculator,
    CoverageReport,
    DataAnalyser,
    DataService,
    PropertyTestRunner,
    TestDatabase,
    generate_ci_config,
)
from .lab_7_03_project_scaffolder import (
    ProjectConfig,
    ProjectScaffolder,
)

__all__ = [
    # Lab 1: Reproducibility
    "ReproducibilityConfig",
    "set_all_seeds",
    "compute_file_hash",
    "verify_file_hash",
    "DataManifest",
    "ExperimentConfig",
    "ExperimentResult",
    "Experiment",
    "timed",
    "retry",
    "generate_readme",
    # Lab 2: Testing and CI/CD
    "Calculator",
    "DataService",
    "DataAnalyser",
    "PropertyTestRunner",
    "TestDatabase",
    "CoverageReport",
    "generate_ci_config",
    # Lab 3: Project Scaffolder
    "ProjectConfig",
    "ProjectScaffolder",
]

__version__ = "7.0.0"
__author__ = "Antonio Clim"
