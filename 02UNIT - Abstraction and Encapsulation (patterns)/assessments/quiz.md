# 02UNIT Quiz: Abstraction and Encapsulation

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Total Questions** | 10 |
| **Multiple Choice** | 6 |
| **Short Answer** | 4 |
| **Time Limit** | 20 minutes |
| **Passing Score** | 70% |

---

## Part A: Multiple Choice (6 questions, 6 marks)

Select the **best** answer for each question.

---

### Question 1

Which of the following best describes the difference between a Python `Protocol` and an `ABC` (Abstract Base Class)?

A) Protocols require explicit inheritance; ABCs use structural typing  
B) Protocols use structural typing; ABCs require explicit inheritance  
C) Protocols can only define methods; ABCs can also define attributes  
D) There is no practical difference between them  

---

### Question 2

Consider the SOLID principle of **Dependency Inversion**. Which statement is correct?

A) High-level modules should depend on low-level modules  
B) Abstractions should depend on details  
C) High-level modules should depend on abstractions, not concrete implementations  
D) Low-level modules should be modified to match high-level requirements  

---

### Question 3

In the **Strategy pattern**, what is the primary benefit of separating the algorithm from the context?

A) It reduces memory usage  
B) It allows algorithms to be swapped at runtime without modifying the context  
C) It improves execution speed  
D) It ensures type safety at compile time  

---

### Question 4

Which Python feature allows you to define a generic class that works with any type?

A) `@dataclass`  
B) `Protocol`  
C) `TypeVar` and `Generic`  
D) `@abstractmethod`  

---

### Question 5

What is the main purpose of the **Observer pattern**?

A) To create objects without specifying their concrete class  
B) To define a one-to-many dependency so when one object changes state, all dependents are notified  
C) To provide a simplified interface to a complex subsystem  
D) To encapsulate a request as an object  

---

### Question 6

Consider this code:

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

What does `frozen=True` accomplish?

A) The Point can only store integer values  
B) The Point instance is immutable and hashable  
C) The Point class cannot be subclassed  
D) The Point fields have default values of 0  

---

## Part B: Short Answer (4 questions, 14 marks)

Answer each question in 2-4 sentences.

---

### Question 7 (3 marks)

Explain the **Liskov Substitution Principle** and give one example of how violating it could cause problems in research software.

---

### Question 8 (3 marks)

Describe the difference between **composition** and **inheritance**. When would you prefer composition over inheritance in designing a simulation framework?

---

### Question 9 (4 marks)

The **Factory pattern** is useful for creating objects. Write pseudocode or Python code showing how you would implement a factory that creates different types of numerical integrators (e.g., Euler, RK4) based on a configuration parameter.

---

### Question 10 (4 marks)

Explain what **type erasure** means in the context of Python's runtime behaviour with generics. How does this differ from languages like Java or C++? Why does this matter for scientific computing?

---

## Answers

<details>
<summary>Click to reveal answers</summary>

### Part A Answers

1. **B** — Protocols use structural typing (duck typing with type hints); ABCs require explicit inheritance.

2. **C** — Dependency Inversion states that high-level modules should depend on abstractions, not concrete low-level implementations.

3. **B** — The Strategy pattern's primary benefit is allowing algorithms to be interchanged at runtime without changing the context class.

4. **C** — `TypeVar` and `Generic` are used to create generic classes that can work with any type while maintaining type safety.

5. **B** — The Observer pattern defines a one-to-many dependency between objects, enabling automatic notification of state changes.

6. **B** — `frozen=True` makes the dataclass immutable (fields cannot be modified after creation) and automatically makes it hashable.

### Part B Answers

7. **Liskov Substitution Principle (LSP)**: Objects of a superclass should be replaceable with objects of a subclass without altering program correctness. A violation example: if a `Bird` class has a `fly()` method and `Penguin` inherits from `Bird` but raises an exception in `fly()`, code expecting any `Bird` to fly will break. In research software, this could occur if a base `Simulation` class promises a `step()` method that advances time, but a subclass overrides it to do something incompatible.

8. **Composition vs Inheritance**: Inheritance creates an "is-a" relationship (a `Dog` is an `Animal`), whilst composition creates a "has-a" relationship (a `Car` has an `Engine`). Prefer composition when you need flexibility to change behaviour at runtime or when inheritance hierarchies become deep and rigid. For simulations, composition allows you to swap integration methods, visualisers or data loggers without modifying the core simulation class.

9. **Factory pseudocode**:
```python
class IntegratorFactory:
    def create(self, method: str, dt: float) -> Integrator:
        match method:
            case "euler":
                return EulerIntegrator(dt)
            case "rk4":
                return RK4Integrator(dt)
            case _:
                raise ValueError(f"Unknown method: {method}")
```

10. **Type erasure**: In Python, generic type information exists only at type-checking time (with tools like mypy) but is erased at runtime. A `list[int]` and `list[str]` are both just `list` at runtime. Java performs similar erasure, but C++ templates generate separate code for each type (monomorphisation). This matters in scientific computing because you cannot inspect generic types at runtime for dynamic dispatch or serialisation without additional metadata.

</details>

---

© 2025 Antonio Clim. All rights reserved.
