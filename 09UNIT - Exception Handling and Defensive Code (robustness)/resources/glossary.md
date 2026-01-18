# 09UNIT: Glossary

## Exception Handling and Defensive Programming

*Definitions of key terms and concepts*

---

### Assertion

A debugging aid that tests a condition and raises `AssertionError` if the condition is false. Assertions verify assumptions during development but can be disabled in production with the `-O` flag. Use for internal invariants, not input validation.

**Example**: `assert len(data) > 0, "Data cannot be empty"`

---

### BaseException

The root of Python's exception hierarchy. All exception classes inherit from `BaseException`. User code should generally catch `Exception` (a subclass) rather than `BaseException` to avoid interfering with system-level exceptions like `KeyboardInterrupt` and `SystemExit`.

---

### Catastrophic Cancellation

A numerical phenomenon where subtracting two nearly equal floating-point numbers produces a result dominated by rounding errors. The significant digits cancel, leaving only noise. Mitigated by reformulating algorithms to avoid subtraction of similar values.

**Example**: Computing `1.0000001 - 1.0` may yield imprecise results.

---

### Circuit Breaker

A design pattern that prevents cascading failures by "opening" (failing fast) when an external service becomes unreliable. The circuit has three states: CLOSED (normal), OPEN (rejecting calls) and HALF_OPEN (testing recovery). Named after electrical circuit breakers that prevent damage from power surges.

---

### Context Manager

An object that defines runtime context for a `with` statement through `__enter__` and `__exit__` methods. Context managers ensure deterministic resource cleanup regardless of how the controlled block exits. Common uses include file handling, database transactions and locks.

**Protocol**: `__enter__(self)` returns resource; `__exit__(self, exc_type, exc_val, exc_tb)` performs cleanup.

---

### Contract (Design by Contract)

A specification of software component behaviour through preconditions, postconditions and invariants. Originated by Bertrand Meyer for the Eiffel language. Contracts define clear responsibilities: callers must satisfy preconditions; functions must establish postconditions; classes must maintain invariants.

---

### Defensive Programming

A programming approach that anticipates and handles potential errors and edge cases to create more resilient code. Includes input validation, boundary checking, assertions and graceful error handling. Particularly valuable in research contexts where data may contain anomalies.

---

### Exception

An object representing an abnormal condition that disrupts normal program flow. Exceptions are raised (thrown) when errors occur and caught (handled) by exception handlers. Python's exception system enables separation of error handling from normal code logic.

---

### Exception Chaining

The mechanism by which one exception can reference another as its cause. Explicit chaining uses `raise ... from ...` and stores the cause in `__cause__`. Implicit chaining occurs when raising during exception handling, stored in `__context__`. Preserves diagnostic information for debugging.

---

### Exception Handler

A code block that responds to a raised exception. In Python, handlers are defined using `except` clauses within `try` statements. Handlers can catch specific exception types or type hierarchies, enabling differentiated responses to different error conditions.

---

### Exception Propagation

The process by which unhandled exceptions "bubble up" through the call stack until caught by a handler or reaching the top level (causing program termination). Also called stack unwinding. The traceback records the path from the exception source.

---

### Exponential Backoff

A retry strategy where the delay between attempts increases exponentially (e.g., 1s, 2s, 4s, 8s). Reduces load on struggling services by spreading retry attempts over time. Often combined with jitter (random variation) to prevent synchronised retries from multiple clients.

---

### Fail-Fast

A design principle advocating immediate, visible failure when errors are detected rather than continuing with potentially corrupt state. Fail-fast systems detect problems early, close to their source, simplifying debugging and preventing cascading failures.

---

### finally Clause

A block in a `try` statement that executes regardless of whether an exception occurred. Used for cleanup operations that must always run. Executes even when the `try` block returns, breaks or continues. The guaranteed execution makes `finally` ideal for resource cleanup.

---

### Graceful Degradation

A design approach where systems continue operating with reduced functionality when components fail, rather than failing completely. For example, a web application might serve cached content when the database is unavailable. Contrasts with fail-fast for user-facing systems.

---

### Guard Clause

An early return or exception at the start of a function that handles edge cases before the main logic. Reduces nesting and makes the "happy path" clearer. Often used to validate inputs and exit early on invalid conditions.

**Example**: `if not data: raise ValueError("Empty data")`

---

### Invariant

A condition that must always be true for an object. Class invariants hold after construction and after every public method call. Loop invariants hold before and after each iteration. Invariants are key to reasoning about program correctness.

---

### Jitter

Random variation added to retry delays to prevent synchronised retries. Without jitter, multiple clients experiencing simultaneous failures would retry at exactly the same moments, creating "thundering herd" effects that overwhelm recovering services.

---

### Postcondition

A condition that a function guarantees to establish upon successful completion, assuming its preconditions were met. Postconditions describe what the function achieves. If a postcondition fails, the function has a bug.

**Example**: "The returned list is sorted in ascending order."

---

### Precondition

A condition that must be true before a function executes. Preconditions are the caller's responsibility. If preconditions are violated, the function's behaviour is undefined. Explicit precondition checking is a key defensive programming technique.

**Example**: "The input list must be non-empty."

---

### RAII (Resource Acquisition Is Initialisation)

A pattern where resource acquisition is tied to object lifetime. Resources are acquired in constructors and released in destructors (or `__exit__`). Python's context managers implement RAII, ensuring deterministic cleanup. Originally from C++.

---

### Retry Pattern

A resilience pattern that automatically re-attempts failed operations. Effective for transient failures (network timeouts, rate limiting). Usually combined with backoff strategies to avoid overwhelming struggling services. Should specify which exception types trigger retries.

---

### Stack Trace (Traceback)

A report of the active stack frames at the time an exception was raised. Shows the sequence of function calls leading to the error, with file names, line numbers and code context. Essential for debugging. Accessed via `traceback` module or `__traceback__` attribute.

---

### Termination Model

The exception handling model used by Python (and most modern languages) where raising an exception transfers control to a handler with no option to resume at the raise point. Contrasts with resumption models where handlers can return control to the exception source.

---

### Thundering Herd

A phenomenon where many clients simultaneously retry or reconnect to a recovering service, overwhelming it and potentially causing another failure. Mitigated by exponential backoff with jitter, which spreads retry attempts over time.

---

### Tolerance (Numerical)

An acceptable margin of error when comparing floating-point values. Relative tolerance (`rel_tol`) scales with value magnitude; absolute tolerance (`abs_tol`) provides a minimum comparison threshold. `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)` implements tolerance-based comparison.

---

### Transient Failure

A temporary failure that resolves itself without intervention, such as network timeouts, rate limiting or brief service unavailability. Transient failures are candidates for retry strategies. Contrast with permanent failures requiring different handling.

---

### Validation

The process of checking that data meets specified requirements. Input validation checks external data before processing; output validation verifies results. Validation may check types, ranges, formats, relationships and domain-specific constraints.

---

*09UNIT: Exception Handling and Defensive Programming*
