# 09UNIT: Self-Assessment Checklist

## Exception Handling and Defensive Programming

---

## Instructions

Complete this self-assessment after finishing the laboratory exercises and before attempting the quiz. For each item, honestly evaluate your current understanding using the scale provided. This reflection helps identify areas requiring additional study.

**Rating Scale:**
- ✅ **Confident**: I can explain this to others and apply it independently
- 🔶 **Developing**: I understand the concept but need more practice
- ❌ **Needs Review**: I need to revisit this topic

---

## §1. Exception Mechanism Fundamentals

| Concept | Rating | Notes |
|---------|--------|-------|
| I can explain the difference between `Exception` and `BaseException` | | |
| I understand why bare `except:` clauses are problematic | | |
| I can order exception handlers from specific to general | | |
| I can use `raise ... from` for exception chaining | | |
| I understand when to use `else` vs putting code in `try` | | |
| I can explain what `__cause__` and `__context__` contain | | |

**Reflection**: What aspect of exception handling do you find most challenging?

> *Your response:*

---

## §2. Custom Exception Hierarchies

| Concept | Rating | Notes |
|---------|--------|-------|
| I can design an exception hierarchy for a specific domain | | |
| I understand when to create custom exceptions vs use built-in ones | | |
| I can add informative attributes to exception classes | | |
| I know to inherit from `Exception`, not `BaseException` | | |
| I can write informative `__str__` methods for exceptions | | |

**Reflection**: Think of a research domain you work in. What custom exceptions might be useful?

> *Your response:*

---

## §3. Context Managers

| Concept | Rating | Notes |
|---------|--------|-------|
| I can implement `__enter__` and `__exit__` methods | | |
| I understand the parameters to `__exit__` | | |
| I know what returning `True` from `__exit__` does | | |
| I can use `@contextmanager` to create context managers | | |
| I understand when to use `ExitStack` | | |
| I can explain the RAII pattern | | |

**Reflection**: What resources in your research code would benefit from context managers?

> *Your response:*

---

## §4. Defensive Programming

| Concept | Rating | Notes |
|---------|--------|-------|
| I can explain preconditions, postconditions and invariants | | |
| I understand the fail-fast principle | | |
| I can implement input validation with informative errors | | |
| I know why to avoid exact floating-point equality | | |
| I can use `math.isclose()` for numerical comparison | | |
| I understand when to use assertions vs exceptions | | |

**Reflection**: How might defensive programming improve your current research code?

> *Your response:*

---

## §5. Resilience Patterns

| Concept | Rating | Notes |
|---------|--------|-------|
| I can implement retry with exponential backoff | | |
| I understand the circuit breaker pattern | | |
| I know when to use graceful degradation | | |
| I can implement checkpoint-based recovery | | |
| I understand the trade-off between fail-fast and graceful degradation | | |

**Reflection**: Which resilience pattern would be most useful for your research workflows?

> *Your response:*

---

## Summary Evaluation

### Strengths

List 2–3 concepts from this unit where you feel most confident:

1. 
2. 
3. 

### Areas for Improvement

List 2–3 concepts requiring additional study:

1. 
2. 
3. 

### Action Plan

What specific steps will you take to address your areas for improvement?

> *Your response:*

---

## Readiness Check

Before proceeding to the quiz, ensure you can answer these questions:

1. **Why should you never use bare `except:` clauses?**

2. **What is the difference between explicit (`raise ... from`) and implicit exception chaining?**

3. **When does the `else` block in a try statement execute?**

4. **What does returning `True` from `__exit__` do?**

5. **What is the fail-fast principle and why is it valuable?**

6. **How do you compare floating-point numbers reliably?**

If you cannot answer any of these confidently, review the corresponding lecture notes section before attempting the quiz.

---

## Submission

Submit this completed self-assessment as part of your unit portfolio. Your honest reflection contributes to the self-assessment grade component (10% of unit grade).

**Date completed**: ________________

**Estimated quiz readiness** (0–100%): ______
