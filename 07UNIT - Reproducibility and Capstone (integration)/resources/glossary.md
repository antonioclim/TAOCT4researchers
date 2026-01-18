# Week 7: Glossary of Terms

> **Terminology reference for reproducibility, testing and research software engineering**

---

## A

### AAA Pattern
**Arrange-Act-Assert** — A standard structure for organising unit tests. First, *arrange* the test data and preconditions; second, *act* by executing the code under test; third, *assert* that the results match expectations.

### Assertion
A statement that checks whether a condition is true. In testing, assertions verify that code behaves as expected. If an assertion fails, the test fails.

```python
assert result == expected, "Optional error message"
```

### Automated Testing
The practice of using software tools to execute tests automatically, rather than running them manually. Enables frequent testing without human intervention.

---

## B

### Benchmarking
The process of measuring and comparing the performance of code, typically in terms of execution time or memory usage. Used to identify bottlenecks and verify performance requirements.

### Build Pipeline
A sequence of automated steps that compile, test and package software. Also known as a build system or build process.

---

## C

### CI/CD
**Continuous Integration / Continuous Deployment** — Practices where code changes are automatically tested (CI) and optionally deployed (CD) when merged to a main branch. Ensures rapid feedback on code quality.

### Code Coverage
A metric indicating the percentage of code executed during testing. Common types include line coverage, branch coverage and function coverage. Higher coverage generally indicates more thorough testing.

### Code Review
The systematic examination of source code by one or more people other than the author. Identifies bugs, improves code quality and shares knowledge across a team.

### conftest.py
A special pytest file that defines fixtures and configuration shared across multiple test files within a directory or project.

### Containerisation
Packaging software and its dependencies into isolated units called containers. Docker is the most common containerisation platform. Ensures consistent environments across different machines.

---

## D

### Data Manifest
A metadata file that documents datasets, including file paths, checksums, sizes and schemas. Used to verify data integrity and track data provenance.

### Dependency
External software that your code requires to function. Dependencies are typically managed through package managers like pip (Python) or npm (JavaScript).

### Dependency Injection
A design pattern where dependencies are provided to a component rather than created within it. Enables testing by allowing mock objects to be substituted.

### Determinism
The property of a system where the same inputs always produce the same outputs. Essential for reproducibility in computational research.

### Docker
A platform for developing, shipping and running applications in containers. Provides environment isolation and reproducibility.

### Dockerfile
A text file containing instructions for building a Docker image. Specifies the base image, dependencies and configuration.

### Docstring
A string literal that documents a module, class or function in Python. Typically placed immediately after the definition.

---

## E

### Edge Case
An unusual or extreme input that may cause unexpected behaviour. Good tests specifically target edge cases to ensure reliability.

### End-to-End Test
A test that verifies the complete flow of an application from start to finish, simulating real user scenarios. Also called E2E tests or system tests.

### Environment
The complete context in which code runs, including the operating system, installed software, environment variables and configuration files.

### Environment Variable
A named value accessible to running processes, used to configure software without modifying code. Examples: `PATH`, `HOME`, `API_KEY`.

---

## F

### FAIR Principles
Guidelines for making data **Findable, Accessible, Interoperable and Reusable**. Widely adopted in scientific research for data management.

### Fixture
In pytest, a function that provides test data or resources. Fixtures enable setup and teardown logic to be shared across tests.

```python
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]
```

### Flaky Test
A test that sometimes passes and sometimes fails without any code changes. Usually caused by race conditions, external dependencies or improper test isolation.

---

## G

### Git
A distributed version control system that tracks changes to files. The most widely used version control system in software development.

### GitHub Actions
A CI/CD platform integrated into GitHub. Enables automated workflows triggered by repository events such as pushes and pull requests.

### Golden File
A reference file containing expected output. Tests compare actual output against the golden file to detect changes. Also called baseline or snapshot.

---

## H

### Hash
A fixed-size value computed from data, used to verify integrity. Common algorithms include SHA-256 and MD5. Changes to data produce different hashes.

### Hook
A mechanism allowing code to be executed at specific points in a process. pytest uses hooks for customising test collection and execution.

---

## I

### Integration Test
A test that verifies the interaction between multiple components or systems. More thorough than unit tests but slower to execute.

### Isolation
The principle that tests should not affect each other. Each test should set up its own preconditions and clean up after itself.

---

## L

### Linter
A tool that analyses code for potential errors, style violations and suspicious constructs. Ruff, Flake8 and Pylint are popular Python linters.

### Lockfile
A file that records the exact versions of all dependencies installed. Examples: `poetry.lock`, `requirements.txt` with pinned versions. Ensures reproducible installations.

---

## M

### Manifest
See **Data Manifest**.

### Matrix Testing
Running the same tests across multiple configurations, such as different Python versions or operating systems. GitHub Actions supports matrix strategies.

### Mock
A test double that simulates the behaviour of real objects. Used to isolate the code under test from external dependencies.

```python
from unittest.mock import Mock
mock_api = Mock(return_value={'status': 'ok'})
```

### Monkey Patching
Dynamically modifying code at runtime, often used in testing to replace functions or objects. pytest provides `monkeypatch` fixture for this purpose.

---

## N

### Nondeterminism
The property of a system where the same inputs may produce different outputs. Common sources include random number generation, timestamps and concurrent execution.

---

## P

### Parametrised Test
A test that runs multiple times with different inputs. Reduces code duplication when testing similar scenarios.

```python
@pytest.mark.parametrize("input,expected", [(1, 1), (2, 4), (3, 9)])
def test_square(input, expected):
    assert square(input) == expected
```

### Pipeline
A sequence of processing stages, often automated. In CI/CD, a pipeline defines the steps from code commit to deployment.

### Pinning
Specifying exact versions of dependencies to ensure reproducible installations. Example: `numpy==1.24.3` instead of `numpy>=1.24`.

### Property-Based Testing
A testing approach where properties that should always hold are defined, and the testing framework generates random inputs to verify them. Hypothesis is a popular Python library.

### Provenance
The documented history of data, including its origin, transformations and custody chain. Essential for reproducibility and audit trails.

### pytest
A popular Python testing framework known for its simplicity and powerful features including fixtures, parametrisation and plugins.

### pyproject.toml
A configuration file for Python projects, defined in PEP 518/517. Contains project metadata, dependencies and tool configuration.

---

## R

### Random Seed
A value that initialises a pseudo-random number generator, enabling reproducible sequences of "random" numbers.

```python
import random
random.seed(42)  # Always produces same sequence
```

### Refactoring
Restructuring existing code without changing its external behaviour. Often performed to improve readability, reduce complexity or enable new features.

### Regression Test
A test that verifies existing functionality still works after code changes. Prevents the reintroduction of bugs.

### Replicability
The ability to obtain consistent results using the same methods but different data, code or implementation. Distinct from reproducibility.

### Reproducibility
The ability to obtain identical results using the same data, code and computational environment. A foundation of scientific validity.

### Research Software Engineering (RSE)
The application of software engineering practices to research contexts. RSEs specialise in developing reliable, maintainable research software.

---

## S

### Scaffolding
Automatically generating the basic structure of a project, including directories, configuration files and templates. Tools like Cookiecutter provide scaffolding.

### Seed
See **Random Seed**.

### Shrinking
In property-based testing, the process of finding the minimal input that still causes a test failure. Aids debugging by simplifying failing cases.

### Side Effect
Any observable change that a function makes beyond returning a value, such as modifying global state, writing to files or making network requests.

### Snapshot Testing
Comparing output against a stored reference (snapshot). If the output changes, the test fails until the snapshot is updated.

### Sphinx
A documentation generator for Python projects. Creates HTML, PDF and other formats from reStructuredText or Markdown source files.

### Static Analysis
Analysing code without executing it. Includes type checking (mypy), linting (ruff) and security scanning.

### Stub
A test double that provides predetermined responses. Simpler than mocks, stubs don't verify how they were called.

---

## T

### TDD (Test-Driven Development)
A development practice where tests are written before the code that makes them pass. Cycle: write failing test → write minimal code → refactor.

### Test Double
A generic term for objects that replace real dependencies in tests. Includes mocks, stubs, fakes and spies.

### Test Fixture
See **Fixture**.

### Test Pyramid
A model for balancing test types: many fast unit tests at the base, fewer integration tests in the middle and few slow end-to-end tests at the top.

### Test Suite
A collection of tests, typically organised by functionality or component. Running a test suite executes all contained tests.

### Type Hint
Annotations that indicate the expected types of function parameters and return values. Enable static type checking with tools like mypy.

```python
def add(a: int, b: int) -> int:
    return a + b
```

---

## U

### Unit Test
A test that verifies a small, isolated piece of code, typically a single function or method. Unit tests should be fast and independent.

---

## V

### Validation
Checking that software meets its requirements and serves its intended purpose. Answers: "Are we building the right thing?"

### Verification
Checking that software is built correctly according to specifications. Answers: "Are we building the thing right?"

### Version Control
A system that records changes to files over time, enabling collaboration and history tracking. Git is the dominant version control system.

### Virtual Environment
An isolated Python environment with its own packages, separate from the system Python. Created with `venv` or `virtualenv`.

---

## W

### Workflow
In GitHub Actions, a configurable automated process defined in YAML. Workflows are triggered by events and execute jobs.

### Working Directory
The current directory from which a script or command is executed. Important for relative file paths.

---

## Y

### YAML
**YAML Ain't Markup Language** — A human-readable data serialisation format used for configuration files, including GitHub Actions workflows.

```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
```

---

## Symbols and Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| AAA | Arrange-Act-Assert |
| ABM | Agent-Based Model |
| API | Application Programming Interface |
| CD | Continuous Deployment/Delivery |
| CI | Continuous Integration |
| CLI | Command-Line Interface |
| CSV | Comma-Separated Values |
| DRY | Don't Repeat Yourself |
| E2E | End-to-End |
| FAIR | Findable, Accessible, Interoperable, Reusable |
| JSON | JavaScript Object Notation |
| RSE | Research Software Engineering |
| SHA | Secure Hash Algorithm |
| TDD | Test-Driven Development |
| YAML | YAML Ain't Markup Language |

---

*© 2025 Antonio Clim. All rights reserved.*
