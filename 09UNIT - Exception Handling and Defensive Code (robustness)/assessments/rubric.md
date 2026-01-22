# 09UNIT: Grading Rubric

## Exception Handling and Defensive Programming

---

## Overview

This rubric provides detailed criteria for evaluating student work across all Unit 09 assessments. Each criterion is scored on a four-level scale reflecting the progression from beginning to excellent performance.

---

## Performance Level Definitions

| Level | Score Range | Description |
|-------|-------------|-------------|
| **Excellent** | 90–100% | Demonstrates mastery; exceeds expectations |
| **Proficient** | 70–89% | Meets all requirements competently |
| **Developing** | 50–69% | Partial understanding; gaps present |
| **Beginning** | 0–49% | Significant gaps; does not meet requirements |

---

## Criterion 1: Exception Handling Implementation

*Weight: 25% of total grade*

### Excellent (90–100%)

- Uses specific exception types appropriate to error conditions
- Implements exception chaining with `raise ... from` consistently
- Handlers ordered from most specific to most general
- Never uses bare `except:` clauses
- Exception messages are informative and include relevant context
- Properly uses `else` and `finally` clauses where appropriate

### Proficient (70–89%)

- Uses appropriate exception types in most cases
- Some exception chaining present
- Handler ordering is generally correct
- Avoids bare `except:` in most cases
- Exception messages provide useful information
- Uses `finally` for cleanup when needed

### Developing (50–69%)

- Uses generic exception types (`Exception`) frequently
- Limited or no exception chaining
- Some handler ordering issues
- Occasional bare `except:` usage
- Exception messages lack detail
- Inconsistent use of `finally` for cleanup

### Beginning (0–49%)

- Catches `BaseException` or uses bare `except:`
- No exception chaining
- Handlers ordered incorrectly
- Exception messages missing or unhelpful
- No use of `else` or `finally` clauses
- Exceptions silently swallowed

---

## Criterion 2: Custom Exception Design

*Weight: 20% of total grade*

### Excellent (90–100%)

- Exception hierarchy reflects domain semantics clearly
- Single inheritance from appropriate base class
- Informative attributes capture error context
- Docstrings explain when each exception is raised
- Consistent naming convention (`*Error` suffix)
- Exception granularity balances specificity and simplicity

### Proficient (70–89%)

- Exception hierarchy is logical
- Inherits from `Exception` appropriately
- Some useful attributes present
- Docstrings present for most exceptions
- Naming follows conventions
- Reasonable granularity

### Developing (50–69%)

- Hierarchy exists but structure is unclear
- May inherit from inappropriate base classes
- Few or no informative attributes
- Incomplete docstrings
- Inconsistent naming
- Too few or too many exception types

### Beginning (0–49%)

- No clear hierarchy design
- Inherits from `BaseException` directly
- No attributes beyond message
- Missing docstrings
- Non-standard naming
- Single exception type for all errors

---

## Criterion 3: Context Manager Implementation

*Weight: 20% of total grade*

### Excellent (90–100%)

- Implements both class-based and decorator-based context managers
- `__enter__` and `__exit__` correctly implemented
- Exception information in `__exit__` handled appropriately
- Resources guaranteed to be released (uses `finally` in generators)
- Uses `ExitStack` for dynamic context management where appropriate
- Clear documentation of context manager behaviour

### Proficient (70–89%)

- Implements working context managers
- Protocol methods are correct
- Resources are released on normal exit
- Exception handling in `__exit__` is present
- Documentation describes basic usage

### Developing (50–69%)

- Context managers partially implemented
- Some protocol issues (e.g., wrong return type from `__exit__`)
- Resource release not guaranteed on exception
- Limited exception handling
- Incomplete documentation

### Beginning (0–49%)

- Context manager protocol not understood
- Missing `__enter__` or `__exit__` methods
- Resources leak on exception
- No exception handling consideration
- No documentation

---

## Criterion 4: Defensive Programming Patterns

*Weight: 20% of total grade*

### Excellent (90–100%)

- Implements preconditions, postconditions and invariants
- Input validation is comprehensive with informative errors
- Fail-fast principle applied consistently
- Numerical resilience techniques used appropriately
- Resilience patterns (retry, circuit breaker) implemented correctly
- Design choices documented and justified

### Proficient (70–89%)

- Some contract-based checking present
- Input validation covers main cases
- Generally fails fast on invalid input
- Aware of numerical precision issues
- Basic resilience patterns implemented

### Developing (50–69%)

- Limited contract checking
- Validation misses some cases
- Some silent failures on invalid input
- Numerical issues not addressed
- Incomplete resilience patterns

### Beginning (0–49%)

- No contract-based validation
- Missing input validation
- Accepts invalid input without error
- Numerical precision ignored
- No resilience patterns

---

## Criterion 5: Code Quality

*Weight: 15% of total grade*

### Excellent (90–100%)

- All functions have complete type hints
- Docstrings follow Google style with examples
- Code passes `ruff` and `mypy --strict`
- Uses `logging` module appropriately (no `print`)
- Uses `pathlib.Path` (no hardcoded strings)
- Consistent naming and formatting
- British English spelling throughout

### Proficient (70–89%)

- Type hints present on most functions
- Docstrings present and informative
- Few linting issues
- Mostly uses logging appropriately
- Uses `pathlib` for paths
- Generally consistent style

### Developing (50–69%)

- Incomplete type hints
- Some docstrings missing or brief
- Multiple linting issues
- Mixed `print` and `logging`
- Inconsistent path handling
- Style inconsistencies

### Beginning (0–49%)

- No type hints
- Missing docstrings
- Many linting errors
- Uses `print` for all output
- Hardcoded path strings
- Poor formatting and naming

---

## Assessment-Specific Rubrics

### Laboratory Exercises

| Criterion | Weight |
|-----------|--------|
| Section completion | 40% |
| Code correctness | 30% |
| Code quality | 20% |
| Documentation | 10% |

### Homework Assignment

| Criterion | Weight |
|-----------|--------|
| Functionality | 40% |
| Design quality | 25% |
| Error handling | 20% |
| Code quality | 15% |

### Quiz

| Section | Weight |
|---------|--------|
| Multiple choice | 60% |
| Short answer | 40% |

---

## Feedback Guidelines

When providing feedback, assessors should:

1. **Identify strengths**: Note what the student did well
2. **Be specific**: Reference exact lines or patterns
3. **Explain why**: Connect feedback to principles
4. **Suggest improvements**: Provide actionable guidance
5. **Encourage**: Recognise effort and progress

### Example Feedback

**Strong work:**
> "Your exception hierarchy clearly reflects the domain model. The `DataValidationError` attributes (field, value, constraint) provide excellent debugging context."

**Area for improvement:**
> "Line 45 uses bare `except:` which catches `SystemExit` and `KeyboardInterrupt`. Consider catching `Exception` instead, or better, the specific exceptions you expect."

---

## Grade Calculation

Final unit grade is calculated as:

| Component | Weight |
|-----------|--------|
| Quiz | 30% |
| Laboratory completion | 40% |
| Homework | 20% |
| Self-assessment | 10% |

**Passing threshold**: 70% overall, with minimum 60% on quiz.
