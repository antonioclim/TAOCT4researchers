# 09UNIT: Quiz

## Exception Handling and Defensive Programming

**Duration**: 30 minutes  
**Total Points**: 50  
**Passing Score**: 70% (35 points)

---

## Instructions

- Answer all questions
- Multiple choice questions have exactly one correct answer
- Short answer questions require concise, specific responses
- Code examples should be syntactically correct Python 3.12+

---

## Section A: Multiple Choice (30 points)

*6 questions × 5 points each*

### Q1. Exception Hierarchy

Which statement about Python's exception hierarchy is **correct**?

A) `KeyboardInterrupt` is a subclass of `Exception`

B) Catching `Exception` will also catch `SystemExit`

C) `FileNotFoundError` is a subclass of `OSError`

D) `BaseException` is a subclass of `Exception`

---

### Q2. Exception Handling Syntax

Consider the following code:

```python
try:
    result = process_data(data)
except ValueError:
    log_error("Invalid value")
    raise
except Exception as e:
    log_error(f"Unexpected: {e}")
else:
    save_result(result)
finally:
    cleanup()
```

When does the `else` block execute?

A) When any exception is caught and handled

B) When the `finally` block completes successfully

C) When the `try` block completes without raising an exception

D) When the function returns a value from the `try` block

---

### Q3. Exception Chaining

What is the **primary purpose** of using `raise NewException() from original_exception`?

A) To suppress the original exception from appearing in tracebacks

B) To replace the original exception type with a more specific one

C) To preserve the causal relationship between exceptions for debugging

D) To automatically retry the operation that caused the original exception

---

### Q4. Custom Exception Design

Which design principle is **violated** by the following exception hierarchy?

```python
class DataError(Exception): pass
class ValidationError(DataError): pass
class FileValidationError(ValidationError, OSError): pass
```

A) Exceptions should inherit from `BaseException`

B) Custom exceptions should avoid multiple inheritance

C) Exception names must end with "Error"

D) Exceptions cannot inherit from built-in types

---

### Q5. Context Manager Protocol

In the `__exit__` method of a context manager, what does returning `True` signify?

A) The context manager completed successfully

B) The exception should be suppressed (not propagated)

C) The `__enter__` method should be called again

D) Resources were released without errors

---

### Q6. Defensive Programming

According to the Design by Contract principle, what is a **precondition**?

A) A condition that must be true after a function returns

B) A condition that the caller must ensure before calling a function

C) A condition checked inside the function before each operation

D) A condition that remains true throughout a class instance's lifetime

---

## Section B: Short Answer (20 points)

*4 questions × 5 points each*

### Q7. Exception Handling Pattern (5 points)

Explain why the following code is problematic and provide a corrected version:

```python
try:
    data = fetch_data(url)
    result = process(data)
    save(result)
except:
    print("Something went wrong")
```

**Your answer:**

---

### Q8. Context Manager Implementation (5 points)

Write a context manager using the `@contextmanager` decorator that:
1. Creates a temporary file
2. Yields the file path
3. Deletes the file when the context exits (even if an exception occurred)

**Your answer:**

---

### Q9. Retry Pattern Analysis (5 points)

A retry mechanism uses exponential backoff with `base_delay=1.0` and `exponential_base=2.0`. Calculate the delays before attempts 1, 2, 3 and 4 (ignoring jitter).

**Your answer:**

---

### Q10. Numerical resilience (5 points)

Explain why the following comparison may fail and provide a resilient alternative:

```python
def check_sum(values: list[float]) -> bool:
    return sum(values) == 1.0
```

**Your answer:**

---

## Answer Key

*For instructor use only*

### Section A

| Question | Answer | Explanation |
|----------|--------|-------------|
| Q1 | C | `FileNotFoundError` inherits from `OSError`. `KeyboardInterrupt` inherits directly from `BaseException`, not `Exception`. |
| Q2 | C | The `else` block executes only when the `try` block completes without raising any exception. |
| Q3 | C | Exception chaining with `from` preserves the `__cause__` attribute, enabling debuggers to show the causal chain. |
| Q4 | B | Multiple inheritance in exceptions creates ambiguous handler matching and should be avoided. |
| Q5 | B | Returning `True` from `__exit__` suppresses the exception; returning `False` or `None` propagates it. |
| Q6 | B | A precondition is the caller's obligation—what must be true before invoking the function. |

### Section B

**Q7** (5 points):

Problems:
- Bare `except` catches everything including `SystemExit` and `KeyboardInterrupt` (2 points)
- Silently swallows errors without logging details (1 point)
- Uses `print` instead of proper logging (1 point)

Corrected version (1 point):
```python
try:
    data = fetch_data(url)
    result = process(data)
    save(result)
except ConnectionError as e:
    logger.error("Network error fetching data: %s", e)
    raise
except ValueError as e:
    logger.error("Invalid data: %s", e)
    raise
```

---

**Q8** (5 points):

```python
from contextlib import contextmanager
import tempfile
from pathlib import Path

@contextmanager
def temp_file():
    path = Path(tempfile.mktemp())
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()
```

Criteria:
- Uses `@contextmanager` decorator (1 point)
- Creates temporary file/path (1 point)
- Yields path (1 point)
- Uses `finally` for cleanup (1 point)
- Handles case where file may not exist (1 point)

---

**Q9** (5 points):

- Attempt 1: `1.0 × 2^0 = 1.0` seconds (or 0 if delay is before retry)
- Attempt 2: `1.0 × 2^1 = 2.0` seconds
- Attempt 3: `1.0 × 2^2 = 4.0` seconds
- Attempt 4: `1.0 × 2^3 = 8.0` seconds

Note: The first attempt has no delay; delays occur *between* attempts.

Criteria:
- Correct formula understanding (2 points)
- Correct calculations (2 points)
- Understanding that delay is between attempts (1 point)

---

**Q10** (5 points):

Problems:
- Floating-point representation errors: `0.1 + 0.2 + 0.7` does not equal exactly `1.0` (2 points)
- Direct equality comparison fails for accumulated floating-point values (1 point)

Resilient alternative (2 points):
```python
import math

def check_sum(values: list[float]) -> bool:
    return math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-12)
```

Or using Kahan summation for improved accuracy with many values.

---

## Grading Summary

| Section | Points | Percentage |
|---------|--------|------------|
| Multiple Choice (Q1-Q6) | /30 | |
| Short Answer (Q7-Q10) | /20 | |
| **Total** | **/50** | |

**Pass/Fail**: ≥35 points required to pass
