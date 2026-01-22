# 09UNIT: Cheatsheet

## Exception Handling and Defensive Programming

*Quick reference for exception handling patterns and defensive programming techniques*

---

## Exception Handling Syntax

### Basic try/except

```python
try:
    result = risky_operation()
except SpecificError as e:
    handle_error(e)
except (ErrorA, ErrorB):
    handle_multiple()
except Exception as e:
    handle_any(e)
else:
    # Runs only if no exception
    process_result(result)
finally:
    # Always runs
    cleanup()
```

### Raising Exceptions

```python
# Raise with message
raise ValueError("Invalid input")

# Re-raise current exception
except ValueError:
    log_error()
    raise

# Chain exceptions (preserve cause)
raise NewError("context") from original_error

# Suppress context
raise NewError("message") from None
```

### Exception Attributes

```python
try:
    operation()
except OSError as e:
    print(e.errno)      # Error number
    print(e.strerror)   # Error message
    print(e.filename)   # Associated filename
    print(e.args)       # Tuple of arguments
    print(e.__cause__)  # Chained exception
    print(e.__context__)# Implicit chain
```

---

## Exception Hierarchy (Key Types)

```
BaseException          ← Don't catch this
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception          ← Catch this instead
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   ├── FileExistsError
    │   └── IsADirectoryError
    ├── ValueError
    ├── TypeError
    ├── AttributeError
    ├── StopIteration
    └── RuntimeError
        └── RecursionError
```

---

## Custom Exceptions

```python
class ResearchError(Exception):
    """Domain-specific base exception."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class DataValidationError(ResearchError):
    """Raised when data fails validation."""
    
    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value
```

### Exception with Recovery Hints

```python
class RecoverableError(Exception):
    """Exception that suggests recovery actions."""
    
    def __init__(self, message: str, recovery_hints: list[str]):
        super().__init__(message)
        self.recovery_hints = recovery_hints
```

---

## Context Managers

### Class-Based

```python
class ManagedResource:
    def __enter__(self) -> Self:
        self.acquire()
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.release()
        return False  # Don't suppress exceptions
```

### Decorator-Based

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire()
    try:
        yield resource
    finally:
        release(resource)
```

### ExitStack (Dynamic Contexts)

```python
from contextlib import ExitStack

with ExitStack() as stack:
    files = [stack.enter_context(open(p)) for p in paths]
    # All files closed when stack exits
```

### suppress (Ignore Specific Exceptions)

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove(path)  # No error if file missing
```

---

## Defensive Programming Patterns

### Design by Contract

```python
def process(data: list[float]) -> float:
    """Process data with contracts.
    
    Precondition: data is non-empty
    Postcondition: result >= 0
    Invariant: data remains unchanged
    """
    assert len(data) > 0, "Precondition: non-empty"
    original_len = len(data)
    
    result = compute(data)
    
    assert len(data) == original_len, "Invariant: length preserved"
    assert result >= 0, "Postcondition: non-negative"
    return result
```

### Input Validation

```python
def validate(value: float, *, min_val: float, max_val: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric, got {type(value).__name__}")
    if not min_val <= value <= max_val:
        raise ValueError(f"Value {value} not in [{min_val}, {max_val}]")
    return value
```

### EAFP vs LBYL

```python
# EAFP: Easier to Ask Forgiveness than Permission (Pythonic)
try:
    value = mapping[key]
except KeyError:
    value = default

# LBYL: Look Before You Leap
if key in mapping:
    value = mapping[key]
else:
    value = default
```

---

## Resilience Patterns

### Retry with Backoff

```python
import time, random

def retry(func, max_attempts=3, base_delay=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except TransientError:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            delay *= 0.5 + random.random() * 0.5  # Jitter
            time.sleep(delay)
```

### Circuit Breaker States

```
CLOSED ──(failures ≥ threshold)──► OPEN
   ▲                                  │
   │                           (timeout)
   │                                  ▼
   └───(successes ≥ threshold)─── HALF_OPEN
```

### Checkpoint Recovery

```python
def process_with_checkpoint(items, checkpoint_file):
    processed = load_checkpoint(checkpoint_file)
    for i, item in enumerate(items):
        if i < processed:
            continue
        result = process_item(item)
        save_checkpoint(checkpoint_file, i + 1)
        yield result
```

---

## Numerical Soundness

### Float Comparison

```python
import math

# DON'T: exact comparison
if a == b:  # Dangerous!

# DO: tolerance-based
if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
    ...
```

### Kahan Summation

```python
def kahan_sum(values):
    total = compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total
```

### Safe Division

```python
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or math.isnan(b):
        return default
    return a / b
```

---

## Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| `except:` | Catches everything | `except Exception:` |
| `except Exception: pass` | Silently swallows | Log and handle |
| `raise Exception(...)` | Loses semantics | Use specific type |
| `== 0.0` for floats | Precision issues | `math.isclose()` |
| No cleanup on error | Resource leaks | Use `finally` or context manager |
| Catch then re-raise differently | Loses traceback | Use `raise ... from` |

---

## Logging with Exceptions

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = operation()
except OperationError:
    logger.exception("Operation failed")  # Includes traceback
    raise
```

---

## Quick Reference

| Pattern | Use When |
|---------|----------|
| `try/except` | Handling expected errors |
| `try/finally` | Guaranteed cleanup |
| `raise ... from` | Preserving error cause |
| Context manager | Resource management |
| Retry with backoff | Transient failures |
| Circuit breaker | External service protection |
| Validation | Input sanitisation |
| Assertions | Development checks |

---

*09UNIT: Exception Handling and Defensive Programming*

© 2025 Antonio Clim. All rights reserved.
