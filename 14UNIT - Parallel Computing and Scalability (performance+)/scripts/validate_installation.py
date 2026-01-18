#!/usr/bin/env python3
"""
14UNIT Installation Validator
Check that all required dependencies are installed.

Usage:
    python scripts/validate_installation.py
"""

import sys
from importlib import import_module
from typing import NamedTuple


class Dependency(NamedTuple):
    name: str
    import_name: str
    required: bool
    min_version: str | None = None


DEPENDENCIES = [
    Dependency("Python", "sys", True, "3.10"),
    Dependency("NumPy", "numpy", True, "1.20"),
    Dependency("Dask", "dask", True, "2023.1"),
    Dependency("Dask Distributed", "distributed", False, None),
    Dependency("Pandas", "pandas", True, "1.3"),
    Dependency("pytest", "pytest", False, None),
    Dependency("line_profiler", "line_profiler", False, None),
    Dependency("memory_profiler", "memory_profiler", False, None),
]


def check_python_version() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    ok = version >= (3, 10)
    return ok, version_str


def check_module(dep: Dependency) -> tuple[bool, str]:
    """Check if module is installed and get version."""
    try:
        mod = import_module(dep.import_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, 'not installed'


def main() -> int:
    print("=" * 60)
    print("14UNIT Installation Validator")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python
    ok, version = check_python_version()
    status = "✓" if ok else "✗"
    print(f"{status} Python: {version} (required: ≥3.10)")
    if not ok:
        all_ok = False
    
    # Check dependencies
    for dep in DEPENDENCIES:
        if dep.name == "Python":
            continue
        
        ok, version = check_module(dep)
        
        if ok:
            status = "✓"
        elif dep.required:
            status = "✗"
            all_ok = False
        else:
            status = "○"  # Optional
        
        req_str = "(required)" if dep.required else "(optional)"
        print(f"{status} {dep.name}: {version} {req_str}")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("All required dependencies are installed. ✓")
        return 0
    else:
        print("Some required dependencies are missing. ✗")
        print("Install with: pip install numpy dask pandas")
        return 1


if __name__ == '__main__':
    sys.exit(main())
