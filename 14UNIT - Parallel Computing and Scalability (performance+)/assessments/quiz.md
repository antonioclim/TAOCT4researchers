# 14UNIT: Quiz

## Parallel Computing and Scalability

**Time Limit**: 20 minutes  
**Passing Score**: 70% (7/10 questions)  
**Format**: 6 Multiple Choice + 4 Short Answer

---

## Section A: Multiple Choice (6 questions)

### Q1. GIL and Threading [LO1]

Which statement best describes Python's Global Interpreter Lock (GIL)?

A) It prevents multiple processes from running simultaneously  
B) It ensures only one thread executes Python bytecode at a time  
C) It locks all global variables during function execution  
D) It prevents race conditions in all Python programs

---

### Q2. Parallelism vs Concurrency [LO1]

A web server handling 1000 simultaneous connections on a single-core processor demonstrates:

A) True parallelism  
B) Concurrency without parallelism  
C) Neither parallelism nor concurrency  
D) Both parallelism and concurrency

---

### Q3. Amdahl's Law [LO1]

If 80% of a program can be parallelised, what is the maximum theoretical speedup according to Amdahl's Law?

A) 4×  
B) 5×  
C) 8×  
D) 80×

---

### Q4. Appropriate Strategy [LO3]

For a task that downloads 100 files from a network server, which approach is most appropriate?

A) multiprocessing.Pool with 100 workers  
B) threading.Thread for each download  
C) Sequential execution with requests  
D) Running 100 separate Python processes

---

### Q5. Race Conditions [LO2, LO3]

What is the primary purpose of using `multiprocessing.Lock`?

A) To speed up parallel execution  
B) To ensure mutual exclusion when accessing shared state  
C) To prevent processes from terminating early  
D) To enable communication between processes

---

### Q6. Dask Schedulers [LO5]

Which Dask scheduler is most appropriate for CPU-bound pure Python computations?

A) synchronous  
B) threads  
C) processes  
D) distributed

---

## Section B: Short Answer (4 questions)

### Q7. Profiling Analysis [LO6]

You profile a function and find that 95% of execution time is spent in a single line that calls `time.sleep(1)`. Explain why parallelisation would be effective here and which approach you would use.

**Your Answer:**

_______________________________________________________________

_______________________________________________________________

_______________________________________________________________

---

### Q8. Pool.map vs Threading [LO2, LO3]

Explain why `multiprocessing.Pool.map` achieves speedup for CPU-bound tasks while `threading` does not, despite both using multiple execution units.

**Your Answer:**

_______________________________________________________________

_______________________________________________________________

_______________________________________________________________

---

### Q9. Speedup Calculation [LO6]

A sequential Monte Carlo simulation takes 120 seconds. The same simulation parallelised across 8 cores takes 20 seconds.

a) Calculate the speedup achieved.  
b) Calculate the parallel efficiency.  
c) Explain why efficiency is less than 100%.

**Your Answer:**

a) Speedup: _______

b) Efficiency: _______

c) Explanation:

_______________________________________________________________

_______________________________________________________________

---

### Q10. Chunking Strategy [LO5]

When creating a Dask array from a 10GB NumPy array on a machine with 4GB RAM, why is choosing an appropriate chunk size critical? What happens if chunks are too large or too small?

**Your Answer:**

_______________________________________________________________

_______________________________________________________________

_______________________________________________________________

---

## Answer Key (Instructor Use Only)

### Multiple Choice

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | B | The GIL is a mutex that allows only one thread to execute Python bytecode at any moment, though it releases during I/O and certain C extensions. |
| 2 | B | A single core cannot execute instructions simultaneously (no parallelism), but can interleave tasks to handle multiple connections (concurrency). |
| 3 | B | S_max = 1/(1-P) = 1/0.2 = 5× |
| 4 | B | Network I/O is I/O-bound; threading is appropriate because the GIL releases during I/O operations. 100 workers would be excessive. |
| 5 | B | Lock provides mutual exclusion, ensuring only one process accesses a critical section at a time. |
| 6 | C | Pure Python CPU-bound work requires processes to bypass the GIL; threads would be serialised. |

### Short Answer Rubric

**Q7** (3 points):
- Identifies I/O-bound nature (sleep releases GIL) [1]
- Recommends threading or asyncio [1]
- Explains that threads can run during sleep [1]

**Q8** (3 points):
- Explains GIL limits threading for CPU-bound work [1]
- Notes multiprocessing uses separate interpreters/GILs [1]
- Correctly distinguishes process isolation vs shared memory [1]

**Q9** (4 points):
- Speedup = 120/20 = 6× [1]
- Efficiency = 6/8 = 75% [1]
- Explains overhead factors (process startup, communication, Amdahl's sequential portion) [2]

**Q10** (3 points):
- Too large: exceeds memory, causes swapping [1]
- Too small: excessive overhead, many small tasks [1]
- Optimal balance: fits in memory while minimising overhead [1]

---

## Grading Summary

| Section | Points | Weight |
|---------|--------|--------|
| Multiple Choice (Q1-Q6) | 6 | 46% |
| Short Answer (Q7-Q10) | 13 | 54% |
| **Total** | **19** | **100%** |

**Passing threshold**: 70% = 13.3 points minimum

---

*14UNIT — Parallel Computing and Scalability*
*Quiz v4.0.0*
