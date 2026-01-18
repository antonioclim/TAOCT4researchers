# 14UNIT: Further Reading

## Parallel Computing and Scalability

---

## Foundational Texts

### Theory and Algorithms

1. **Herlihy, M., & Shavit, N. (2012). *The Art of Multiprocessor Programming* (Revised ed.). Morgan Kaufmann.**
   
   Comprehensive treatment of concurrent programming principles, from basic locks to lock-free algorithms. Essential for understanding synchronisation at a deep level.
   
   *Relevance: LO2, LO3 — synchronisation primitives and concurrent data structures*

2. **McCool, M., Reinders, J., & Robison, A. (2012). *Structured Parallel Programming: Patterns for Efficient Computation*. Morgan Kaufmann.**
   
   Patterns-based approach to parallel algorithm design. Covers map, reduce, scan, fork-join, and pipeline patterns with practical implementations.
   
   *Relevance: LO2 — parallel patterns and algorithm design*

3. **Mattson, T., Sanders, B., & Massingill, B. (2004). *Patterns for Parallel Programming*. Addison-Wesley.**
   
   Design patterns specifically for parallel programming. Complements Gang of Four patterns with concurrency-specific solutions.
   
   *Relevance: LO2, LO4 — high-level parallel design*

### Python-Specific

4. **Gorelick, M., & Ozsvald, I. (2020). *High Performance Python: Practical Performant Programming for Humans* (2nd ed.). O'Reilly Media.**
   
   Python-specific performance optimisation including profiling, Cython, multiprocessing, and clusters. Highly practical with real-world case studies.
   
   *Relevance: LO2, LO6 — Python parallelisation and profiling*

5. **Palach, J. (2014). *Parallel Programming with Python*. Packt Publishing.**
   
   Focused treatment of Python's parallel computing capabilities. Covers threading, multiprocessing, and distributed computing.
   
   *Relevance: LO2, LO3, LO4 — Python parallel programming*

---

## Research Papers

### Foundational Theory

6. **Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*, 30, 483-485.**
   
   The foundational paper establishing theoretical limits on parallel speedup. Essential reading for understanding why adding processors has diminishing returns.
   
   *Relevance: LO1 — theoretical foundations*

7. **Gustafson, J. L. (1988). Reevaluating Amdahl's law. *Communications of the ACM*, 31(5), 532-533.**
   
   The counterpoint to Amdahl, arguing that problem size scales with available resources. Introduces "scaled speedup" concept.
   
   *Relevance: LO1 — theoretical foundations*

8. **Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified data processing on large clusters. *Communications of the ACM*, 51(1), 107-113.**
   
   The influential paper introducing MapReduce paradigm at Google. Foundation for modern distributed computing frameworks.
   
   *Relevance: LO5 — distributed computing concepts*

### Python and GIL

9. **Beazley, D. (2010). Understanding the Python GIL. *PyCon 2010*. [Video and slides]**
   
   Technical examination of GIL mechanics by a Python expert. Essential for understanding why threading behaves as it does in Python.
   
   *Relevance: LO1, LO3 — GIL understanding*
   
   URL: https://www.dabeaz.com/python/UnderstandingGIL.pdf

10. **Beazley, D. (2009). An Inside Look at the GIL Removal Patch. *PyCon 2009*.**
    
    Explores why removing the GIL is difficult and what trade-offs are involved. Illuminates deep interpreter design issues.
    
    *Relevance: LO1 — GIL architecture*

---

## Online Documentation

### Python Standard Library

11. **Python Documentation: multiprocessing — Process-based parallelism**
    
    Official documentation for the multiprocessing module. Authoritative reference for all classes and functions.
    
    URL: https://docs.python.org/3/library/multiprocessing.html
    
    *Relevance: LO2 — multiprocessing API*

12. **Python Documentation: concurrent.futures — Launching parallel tasks**
    
    Official documentation for the high-level parallel execution interface.
    
    URL: https://docs.python.org/3/library/concurrent.futures.html
    
    *Relevance: LO4 — concurrent.futures API*

13. **Python Documentation: threading — Thread-based parallelism**
    
    Official threading documentation. Important for understanding Python's threading model.
    
    URL: https://docs.python.org/3/library/threading.html
    
    *Relevance: LO3 — threading API*

### Dask Framework

14. **Dask Documentation**
    
    Comprehensive official documentation for Dask. Includes tutorials, API reference, and recommended approaches.
    
    URL: https://docs.dask.org/
    
    *Relevance: LO5 — Dask usage*

15. **Dask Tutorial (Official)**
    
    Interactive tutorial covering Dask fundamentals with Jupyter notebooks.
    
    URL: https://tutorial.dask.org/
    
    *Relevance: LO5 — hands-on Dask learning*

---

## Tutorials and Courses

### Online Tutorials

16. **Real Python: Speed Up Your Python Program With Concurrency**
    
    Practical tutorial comparing threading, multiprocessing, and asyncio with clear examples.
    
    URL: https://realpython.com/python-concurrency/
    
    *Relevance: LO2, LO3 — practical Python parallelism*

17. **Real Python: Python Multiprocessing: The Complete Guide**
    
    In-depth multiprocessing tutorial with practical examples.
    
    URL: https://realpython.com/python-multiprocessing/
    
    *Relevance: LO2 — multiprocessing details*

18. **Coiled: Dask Guidelines**
    
    Production-oriented guide to using Dask effectively from the company behind Dask.
    
    URL: https://docs.coiled.io/user_guide/best-practices.html
    
    *Relevance: LO5 — production Dask usage*

---

## Tools and Utilities

### Profiling

19. **line_profiler Documentation**
    
    Line-by-line profiling for Python. Essential for identifying exact bottleneck locations.
    
    URL: https://github.com/pyutils/line_profiler
    
    *Relevance: LO6 — detailed profiling*

20. **memory_profiler Documentation**
    
    Memory profiling for Python functions. Tracks memory allocation over time.
    
    URL: https://github.com/pythonprofilers/memory_profiler
    
    *Relevance: LO6 — memory analysis*

### Visualisation and Monitoring

21. **Dask Dashboard**
    
    Real-time monitoring interface for Dask computations. Shows task progress, memory usage, and worker status.
    
    URL: https://docs.dask.org/en/latest/diagnostics-distributed.html
    
    *Relevance: LO5 — Dask monitoring*

---

## Advanced Topics

### Distributed Computing

22. **Dask Distributed Documentation**
    
    Documentation for scaling Dask across multiple machines.
    
    URL: https://distributed.dask.org/
    
    *Relevance: LO5 (extension) — distributed scaling*

23. **Ray Documentation**
    
    Alternative distributed computing framework with different trade-offs than Dask.
    
    URL: https://docs.ray.io/
    
    *Relevance: LO5 (extension) — alternative frameworks*

### GPU Computing

24. **RAPIDS Documentation**
    
    GPU-accelerated data science libraries compatible with Dask.
    
    URL: https://rapids.ai/
    
    *Relevance: LO5 (extension) — GPU acceleration*

---

## Reading Schedule Recommendation

| Week | Focus | Resources |
|------|-------|-----------|
| 1 | Theory | 6, 7, 9 |
| 2 | multiprocessing | 4 (Ch. 9), 11, 16 |
| 3 | Threading/Futures | 12, 13, 17 |
| 4 | Dask | 14, 15, 18 |
| 5 | Profiling | 4 (Ch. 2), 19, 20 |
| 6+ | Advanced | 1, 2, 22, 23 |

---

## Citation Format

When referencing these materials in academic work, use appropriate citation format:

**Book**:
> Gorelick, M., & Ozsvald, I. (2020). *High Performance Python* (2nd ed.). O'Reilly Media.

**Paper**:
> Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*, 30, 483-485.

**Online**:
> Python Software Foundation. (2024). *multiprocessing — Process-based parallelism*. Python 3.12 Documentation. https://docs.python.org/3/library/multiprocessing.html

---

*14UNIT — Parallel Computing and Scalability*
*Further Reading v4.0.0*
