# 09UNIT: Further Reading

## Exception Handling and Defensive Programming

*Annotated bibliography for deeper exploration of unit concepts*

---

## Core References

### Python Documentation

**Python Documentation: Errors and Exceptions**  
https://docs.python.org/3/tutorial/errors.html

The official tutorial provides a comprehensive introduction to Python's exception handling syntax and built-in exception types. necessary reading for understanding the language's approach to error management. The accompanying reference documentation details the complete exception hierarchy.

---

**Python Documentation: Context Manager Types**  
https://docs.python.org/3/library/stdtypes.html#context-manager-types

Official documentation of the context manager protocol including `__enter__` and `__exit__` semantics. Explains exception handling in `__exit__` and the `contextlib` module's utilities for creating context managers.

---

### Foundational Texts

**Beazley, D. (2009). *Python necessary Reference* (4th ed.). Addison-Wesley.**

Chapter 5 provides detailed coverage of Python's exception mechanism, including exception attributes, chaining and conventions. Beazley's treatment emphasises practical application in production systems. The book remains valuable despite the emergence of newer Python versions.

---

**Meyer, B. (1997). *Object-Oriented Software Construction* (2nd ed.). Prentice Hall.**

The definitive treatment of design by contract from its creator. Meyer's book explains the philosophical foundation of contracts and their role in software correctness. Chapters 11-12 cover preconditions, postconditions and invariants in depth. necessary for understanding defensive programming's theoretical basis.

---

**Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns*. Addison-Wesley.**

The "Gang of Four" book includes patterns relevant to exception handling and resource management. The Template Method and Strategy patterns inform flexible error handling designs. Understanding these patterns enhances ability to design comprehensive exception hierarchies.

---

## Research and Practice

### Software Engineering

**Nystrom, R. (2021). *Crafting Interpreters*. Genever Benning.**  
https://craftinginterpreters.com/

Chapter 9 covers error handling in language implementation. While focused on interpreter design, the principles of error detection, reporting and recovery apply broadly. Freely available online with excellent visualisations.

---

**McConnell, S. (2004). *Code Complete* (2nd ed.). Microsoft Press.**

Chapter 8 addresses defensive programming practices including assertions, error handling and exception guidelines. McConnell provides practical heuristics for deciding when to use exceptions versus error codes. Industry-focused but applicable to research software.

---

**Hunt, A., & Thomas, D. (2019). *The Pragmatic Programmer* (20th Anniversary ed.). Addison-Wesley.**

Topic 23 "Design by Contract" and Topic 24 "Dead Programs Tell No Lies" (fail-fast) provide accessible introductions to defensive programming. The authors emphasise crash-early debugging benefits and contract-based design thinking.

---

### Scientific Computing

**Downey, A. B. (2015). *Think Python* (2nd ed.). O'Reilly Media.**  
https://greenteapress.com/wp/think-python-2e/

Chapters on debugging and exception handling are oriented toward scientific applications. Downey discusses defensive programming in data transformation contexts relevant to research computing. Freely available online.

---

**Wilson, G., et al. (2017). "Good Enough Practices in Scientific Computing." *PLOS Computational Biology*, 13(6), e1005510.**  
https://doi.org/10.1371/journal.pcbi.1005510

Practical recommendations for research software development including error handling, testing and documentation. The "good enough" framing is particularly relevant for researchers balancing correctness with productivity.

---

**Wilson, G., et al. (2014). "conventions for Scientific Computing." *PLOS Biology*, 12(1), e1001745.**  
https://doi.org/10.1371/journal.pbio.1001745

Broader coverage of scientific computing practices with sections on defensive programming and error handling. Provides context for why these practices matter in research environments.

---

## Specialised Topics

### Resilience Patterns

**Nygard, M. T. (2018). *Release It!* (2nd ed.). Pragmatic Bookshelf.**

The definitive treatment of resilience patterns for production systems. Chapters on stability patterns cover circuit breakers, bulkheads, timeouts and retry strategies in depth. While enterprise-focused, the patterns apply to any system interacting with external services.

---

**Fowler, M. (2014). "Circuit Breaker." *martinfowler.com*.**  
https://martinfowler.com/bliki/CircuitBreaker.html

Accessible introduction to the circuit breaker pattern with implementation guidance. Fowler explains the pattern's motivation and state machine clearly. Good starting point before Nygard's more comprehensive treatment.

---

### Numerical Computing

**Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*, 23(1), 5-48.**

The classic reference on floating-point representation and arithmetic. Explains the sources of numerical error and defensive practices for numerical code. Dense but necessary for anyone doing scientific computation.

---

**Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.**

Comprehensive treatment of numerical stability in algorithm design. Relevant for understanding when defensive practices like Kahan summation are necessary. Graduate-level but accessible chapters on error analysis fundamentals.

---

### Python-Specific

**Ramalho, L. (2022). *Fluent Python* (2nd ed.). O'Reilly Media.**

Chapter 18 covers context managers and the `with` statement comprehensively. Ramalho explains the protocol in detail and demonstrates advanced patterns including `ExitStack`. necessary for Python practitioners.

---

**Hettinger, R. (2013). "Transforming Code into Beautiful, Idiomatic Python." PyCon US 2013.**  
https://www.youtube.com/watch?v=OSGv2VnC0go

This presentation includes discussion of Python's exception handling idioms and the EAFP (Easier to Ask Forgiveness than Permission) philosophy. Hettinger demonstrates pythonic error handling patterns.

---

## Online Resources

### Documentation and Tutorials

**Real Python: Exception Handling**  
https://realpython.com/python-exceptions/

Comprehensive tutorial covering Python exception handling from basics through advanced patterns. Includes interactive examples and exercises. Good supplementary material for beginners.

---

**Python Exception Hierarchy Diagram**  
https://docs.python.org/3/library/exceptions.html#exception-hierarchy

Visual representation of the complete built-in exception hierarchy. Useful reference when designing custom exception hierarchies.

---

### Standards and Guidelines

**PEP 343: The "with" Statement**  
https://peps.python.org/pep-0343/

The Python Enhancement Proposal that introduced context managers. Explains the design rationale and protocol specification. Understanding the PEP deepens appreciation of context manager mechanics.

---

**PEP 3134: Exception Chaining and Embedded Tracebacks**  
https://peps.python.org/pep-3134/

Specification for exception chaining introduced in Python 3. Explains `__cause__` and `__context__` attributes and the `raise ... from ...` syntax. necessary for understanding exception chain semantics.

---

**PEP 654: Exception Groups and except***  
https://peps.python.org/pep-0654/

Python 3.11 introduced exception groups for concurrent exception handling. Relevant for advanced error handling in async and parallel contexts. Forward-looking reading for modern Python development.

---

## Historical Context

**Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice Hall.**

Foundational text on program correctness including preconditions and postconditions. Dijkstra's work on structured programming informs modern defensive programming practices. Theoretical but influential.

---

**Liskov, B., & Snyder, A. (1979). "Exception Handling in CLU." *IEEE Transactions on Software Engineering*, SE-5(6), 546-558.**

Describes the termination model exception handling that influenced Python. Understanding CLU's design choices illuminates why Python exceptions work as they do.

---

*09UNIT: Exception Handling and Defensive Programming*
