# Week 3: Further Reading

> **Curated Resources for Algorithmic Complexity**  
> © 2025 Antonio Clim. All rights reserved.

---

## 📚 Textbooks

### Essential

1. **Introduction to Algorithms** (4th Edition)  
   *Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein*  
   MIT Press, 2022  
   ISBN: 978-0262046305  
   
   The definitive textbook on algorithms. Chapters 2-4 cover growth of functions, recurrences and probabilistic analysis. Chapter 17 covers amortised analysis in depth.
   
   **Relevant Chapters:**
   - Chapter 2: Getting Started (insertion sort analysis)
   - Chapter 3: Growth of Functions (Big-O, Big-Θ, Big-Ω)
   - Chapter 4: Divide-and-Conquer (Master Theorem)
   - Chapter 17: Amortised Analysis

2. **Algorithm Design**  
   *Jon Kleinberg, Éva Tardos*  
   Pearson, 2005  
   ISBN: 978-0321295354
   
   Excellent for algorithm design techniques with clear complexity analysis. Strong focus on graph algorithms and network flows.

3. **The Art of Computer Programming, Volume 1: Fundamental Algorithms**  
   *Donald E. Knuth*  
   Addison-Wesley, 3rd Edition, 1997  
   ISBN: 978-0201896831
   
   The classic reference. Mathematically rigorous treatment of algorithm analysis. Section 1.2.11 covers asymptotic representations.

### Supplementary

4. **Algorithms** (4th Edition)  
   *Robert Sedgewick, Kevin Wayne*  
   Addison-Wesley, 2011  
   ISBN: 978-0321573513
   
   Practical approach with extensive Java implementations. Excellent visualisations of algorithm behaviour. Companion website: [algs4.cs.princeton.edu](https://algs4.cs.princeton.edu/)

5. **Grokking Algorithms**  
   *Aditya Bhargava*  
   Manning, 2016  
   ISBN: 978-1617292231
   
   Beginner-friendly illustrated guide. Good for intuition building before diving into formal analysis.

---

## 📄 Research Papers

6. **Amortised Computational Complexity**  
   *Robert E. Tarjan*  
   SIAM Journal on Algebraic and Discrete Methods, 1985  
   DOI: 10.1137/0606031
   
   The seminal paper introducing amortised analysis. Defines the potential method and applies it to self-adjusting data structures.

7. **Cache-Oblivious Algorithms**  
   *Matteo Frigo, Charles E. Leiserson, Harald Prokop, Sridhar Ramachandran*  
   FOCS 1999  
   DOI: 10.1109/SFFCS.1999.814600
   
   Introduces cache-oblivious algorithms that automatically adapt to memory hierarchy without knowing cache parameters.

8. **The Analysis of Algorithms**  
   *Donald E. Knuth*  
   Proceedings of the International Congress of Mathematicians, 1970
   
   Historical paper establishing the mathematical foundations of algorithm analysis.

---

## 🌐 Online Courses

9. **MIT OpenCourseWare: Introduction to Algorithms (6.006)**  
   Massachusetts Institute of Technology  
   [ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)
   
   Free access to lecture videos, notes and problem sets from MIT's undergraduate algorithms course. Lectures 1-3 cover asymptotic complexity.

10. **Coursera: Algorithms Specialisation**  
    *Tim Roughgarden, Stanford University*  
    [coursera.org/specializations/algorithms](https://www.coursera.org/specializations/algorithms)
    
    Four-course sequence covering divide-and-conquer, graph algorithms, greedy algorithms and dynamic programming. Strong emphasis on complexity analysis.

11. **Khan Academy: Algorithms**  
    [khanacademy.org/computing/computer-science/algorithms](https://www.khanacademy.org/computing/computer-science/algorithms)
    
    Free, beginner-friendly introduction with interactive exercises. Good coverage of Big-O notation and sorting algorithms.

---

## 🛠️ Tools and Libraries

12. **timeit (Python Standard Library)**  
    [docs.python.org/3/library/timeit.html](https://docs.python.org/3/library/timeit.html)
    
    Python's built-in module for timing small code snippets. Handles warmup and multiple iterations automatically.
    
    ```python
    import timeit
    timeit.timeit('sorted(range(1000))', number=1000)
    ```

13. **memory_profiler**  
    [pypi.org/project/memory-profiler](https://pypi.org/project/memory-profiler/)
    
    Line-by-line memory usage profiling for Python. Essential for space complexity analysis.
    
    ```bash
    pip install memory_profiler
    python -m memory_profiler script.py
    ```

14. **big_O Library**  
    [github.com/pberkes/big_O](https://github.com/pberkes/big_O)
    
    Automatic empirical complexity estimation for Python functions. Fits measured data to standard complexity classes.
    
    ```python
    from big_o import big_o, datagen
    positive_int_generator = lambda n: datagen.integers(n, 0, 10000)
    best, others = big_o.big_o(sorted, positive_int_generator, n_repeats=20)
    ```

15. **perf (Linux)**  
    [perf.wiki.kernel.org](https://perf.wiki.kernel.org/)
    
    Linux profiling tool for hardware performance counters. Measures cache misses, branch mispredictions and other low-level metrics.
    
    ```bash
    perf stat -e cache-misses,cache-references python script.py
    ```

16. **Valgrind (Cachegrind)**  
    [valgrind.org/docs/manual/cg-manual.html](https://valgrind.org/docs/manual/cg-manual.html)
    
    Cache profiler simulating CPU caches. Detailed cache miss statistics for understanding memory hierarchy effects.

---

## 📊 Visualisation Resources

17. **VisuAlgo**  
    [visualgo.net](https://visualgo.net/)
    
    Interactive visualisations of data structures and algorithms. See sorting, searching and graph algorithms animated with complexity annotations.

18. **Algorithm Visualiser**  
    [algorithm-visualizer.org](https://algorithm-visualizer.org/)
    
    Open-source platform for visualising algorithms from code. Supports multiple programming languages.

19. **Big-O Cheat Sheet**  
    [bigocheatsheet.com](https://www.bigocheatsheet.com/)
    
    Quick reference for common data structure and algorithm complexities. Includes space complexity.

---

## 📖 Blog Posts and Articles

20. **A Gentle Introduction to Algorithm Complexity Analysis**  
    *Dionysis Zindros*  
    [discrete.gr/complexity](https://discrete.gr/complexity/)
    
    Excellent tutorial starting from basics and building to Master Theorem. Includes interactive examples.

21. **The Log/Log Plot**  
    *Evan Miller*  
    [evanmiller.org/the-log-log-plot.html](https://www.evanmiller.org/the-log-log-plot.html)
    
    How to use log-log plots to empirically determine complexity. Power-law relationships appear as straight lines.

22. **Gallery of Processor Cache Effects**  
    *Igor Ostrovsky*  
    [igoro.com/archive/gallery-of-processor-cache-effects](https://igoro.com/archive/gallery-of-processor-cache-effects/)
    
    Practical demonstrations of cache effects on algorithm performance. Explains why constants matter.

---

## 🎥 Video Resources

23. **3Blue1Brown: But what is the Central Limit Theorem?**  
    [youtube.com/watch?v=zeJD6dqJ5lo](https://www.youtube.com/watch?v=zeJD6dqJ5lo)
    
    While not directly about complexity, understanding distributions is essential for average-case analysis.

24. **MIT OCW: Lecture 1 - Algorithms and Computation**  
    [youtube.com/watch?v=ZA-tUyM_y7s](https://www.youtube.com/watch?v=ZA-tUyM_y7s)
    
    Erik Demaine's introductory lecture on algorithm analysis. Covers the "why" of Big-O notation.

---

## 🔬 Research-Specific Resources

25. **Empirical Algorithmics**  
    *Catherine McGeoch*  
    Communications of the ACM, 2007  
    DOI: 10.1145/1297797.1297816
    
    Methodology for experimental analysis of algorithms. Essential reading for researchers benchmarking algorithms.

26. **Benchmarking Crimes: An Emerging Threat in Systems Security**  
    *Gernot Heiser*  
    ACM SIGOPS Operating Systems Review, 2018
    
    Common mistakes in benchmarking and how to avoid them. Relevant for any researcher measuring performance.

---

## 🧮 Mathematical Background

27. **Concrete Mathematics: A Foundation for Computer Science**  
    *Ronald L. Graham, Donald E. Knuth, Oren Patashnik*  
    Addison-Wesley, 2nd Edition, 1994  
    ISBN: 978-0201558029
    
    Mathematical techniques for algorithm analysis: sums, recurrences, generating functions. Chapter 9 covers asymptotic analysis.

28. **Mathematics for Computer Science**  
    *Eric Lehman, F. Thomson Leighton, Albert R. Meyer*  
    MIT OpenCourseWare, 2018  
    [ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)
    
    Free textbook covering discrete mathematics for CS. Chapter 15 covers cardinality and countability relevant to complexity theory.

---

## 📝 Practice Problems

29. **LeetCode**  
    [leetcode.com](https://leetcode.com/)
    
    Coding challenges with complexity requirements. Filter by difficulty and topic. "Explore" section has complexity-focused learning paths.

30. **Project Euler**  
    [projecteuler.net](https://projecteuler.net/)
    
    Mathematical programming problems. Many require algorithmic insight to solve within reasonable time—brute force will not suffice.

---

## 🔗 Quick Links Summary

| Resource | Type | Difficulty | URL |
|----------|------|------------|-----|
| CLRS | Textbook | Advanced | ISBN: 978-0262046305 |
| MIT 6.006 | Course | Intermediate | ocw.mit.edu |
| VisuAlgo | Visualisation | Beginner | visualgo.net |
| Big-O Cheat Sheet | Reference | All levels | bigocheatsheet.com |
| memory_profiler | Tool | Intermediate | pypi.org |

---

## 📌 Recommended Reading Order

**For Beginners:**
1. Grokking Algorithms (book)
2. Khan Academy Algorithms (online)
3. VisuAlgo (visualisation)
4. Big-O Cheat Sheet (reference)

**For Intermediate Learners:**
1. Algorithms by Sedgewick (book)
2. MIT 6.006 (online course)
3. Discrete.gr Complexity Tutorial (article)
4. LeetCode Easy/Medium problems

**For Advanced Learners:**
1. Introduction to Algorithms (CLRS)
2. Tarjan's Amortised Analysis paper
3. Cache-Oblivious Algorithms paper
4. Concrete Mathematics (for proofs)

---

*For questions or suggestions about these resources, contact the course instructor.*

© 2025 Antonio Clim. All rights reserved.
