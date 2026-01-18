# 14UNIT: Glossary

## Parallel Computing and Scalability

---

## A

**Amdahl's Law**
: Theoretical model predicting maximum parallel speedup. If fraction *P* of a program is parallelisable, maximum speedup is 1/(1-P). Named after Gene Amdahl (1967).

**Asynchronous Execution**
: Execution model where tasks run independently without blocking the caller. Results are retrieved later via callbacks or Future objects.

**Atomic Operation**
: Operation that completes entirely or not at all, with no possibility of interruption. In Python, simple variable assignments are atomic, but read-modify-write operations are not.

---

## B

**Barrier**
: Synchronisation primitive where multiple processes/threads wait until all have reached a common point before any proceeds.

**Blocking Call**
: Function call that suspends execution until the operation completes. Contrasted with non-blocking/asynchronous calls.

---

## C

**Chunk**
: Subdivision of data for parallel processing. In Dask, chunks define the granularity of parallel operations on arrays and DataFrames.

**Concurrency**
: Managing multiple tasks that may overlap in time. Tasks interleave but need not execute simultaneously. Distinct from parallelism.

**concurrent.futures**
: Python standard library module providing high-level interface for asynchronous execution using thread or process pools.

**Context Switch**
: Operating system operation of saving one process/thread's state and loading another's. Frequent context switching adds overhead.

**CPU-Bound**
: Computation limited by processor speed rather than I/O. Benefits from parallel execution across CPU cores.

---

## D

**Dask**
: Python library for parallel computing, providing familiar NumPy/pandas interfaces for larger-than-memory datasets and parallel execution.

**Dask Array**
: Dask's parallel, chunked array implementation compatible with NumPy's API.

**Dask DataFrame**
: Dask's parallel, partitioned DataFrame implementation compatible with pandas' API.

**Deadlock**
: Situation where two or more processes wait indefinitely for resources held by each other. Classic example: Process A holds Lock 1, waits for Lock 2; Process B holds Lock 2, waits for Lock 1.

**Delayed Computation**
: Lazy evaluation pattern where computation is deferred until explicitly requested. Dask's `@delayed` decorator implements this pattern.

---

## E

**Efficiency**
: Ratio of speedup to processor count: E = S/n. Measures how effectively parallel resources are utilised. Ideal efficiency is 1.0 (100%).

**Embarrassingly Parallel**
: Problem class where computations are independent and require no inter-process communication. Examples: Monte Carlo, map operations, parameter sweeps.

**Event**
: Synchronisation primitive for signalling between threads/processes. One process sets the event; others wait for it.

**Executor**
: Object managing a pool of workers (threads or processes) for executing callable objects. See ProcessPoolExecutor, ThreadPoolExecutor.

---

## F

**Fork**
: Process creation method (Unix) that copies parent's memory space. Fast but potentially unsafe with threads. See also: spawn, forkserver.

**Future**
: Object representing the result of an asynchronous computation. Provides methods to check completion, retrieve results, and handle exceptions.

---

## G

**GIL (Global Interpreter Lock)**
: Mutex in CPython ensuring only one thread executes Python bytecode at a time. Simplifies memory management but limits threading for CPU-bound work.

**Gustafson's Law**
: Alternative to Amdahl's Law assuming problem size scales with processor count. Scaled speedup: S(n) = n - α(n-1), where α is serial fraction.

---

## I

**I/O-Bound**
: Computation limited by input/output operations (disk, network, database). Threading provides benefit because GIL releases during I/O.

**Inter-Process Communication (IPC)**
: Mechanisms for data exchange between processes: pipes, queues, shared memory, sockets.

---

## L

**Lazy Evaluation**
: Delaying computation until the result is needed. Enables optimisation and out-of-core processing.

**Load Balancing**
: Distributing work evenly across workers to maximise utilisation and minimise idle time.

**Lock (Mutex)**
: Synchronisation primitive ensuring mutual exclusion. Only one thread/process can hold the lock at a time.

---

## M

**Map-Reduce**
: Parallel pattern: apply function to data partitions (map), then combine results (reduce). Foundation of distributed computing frameworks.

**Memory-Mapped File**
: File whose contents are mapped directly into virtual memory, enabling efficient shared access across processes.

**Monte Carlo Method**
: Computational technique using random sampling to obtain numerical results. Named after the Monte Carlo casino in Monaco.

**Multiprocessing**
: Python module providing process-based parallelism. Each process has its own Python interpreter and GIL.

---

## O

**Out-of-Core Computation**
: Processing data that exceeds available memory by working on chunks sequentially. Dask specialises in this pattern.

---

## P

**Parallelism**
: Simultaneous execution of multiple computations on separate processing units. Requires multiple CPU cores or machines.

**Partition**
: Subdivision of a DataFrame or dataset. Dask DataFrames consist of multiple partitions processed in parallel.

**Pickle**
: Python's serialisation format. Multiprocessing uses pickle to transfer objects between processes. Some objects (lambdas, closures) cannot be pickled.

**Pool**
: Collection of worker processes reused across multiple tasks. Amortises process creation overhead.

**Process**
: Independent execution environment with its own memory space. Heavier than threads but provides isolation and bypasses the GIL.

**Profiling**
: Measuring program execution to identify performance characteristics. Tools: cProfile, line_profiler, memory_profiler.

---

## Q

**Queue**
: Thread/process-safe FIFO data structure for passing objects between concurrent units. multiprocessing.Queue enables IPC.

---

## R

**Race Condition**
: Bug where program behaviour depends on the relative timing of concurrent operations. Occurs when shared state is accessed without proper synchronisation.

**RLock (Reentrant Lock)**
: Lock that can be acquired multiple times by the same thread without deadlock.

---

## S

**Scheduler**
: Component deciding when and where to execute tasks. Dask supports synchronous, threaded, processes, and distributed schedulers.

**Semaphore**
: Synchronisation primitive allowing limited concurrent access. A semaphore initialised to n permits n simultaneous holders.

**Shared Memory**
: Memory region accessible by multiple processes. multiprocessing.Value and Array provide shared memory primitives.

**Spawn**
: Process creation method starting a fresh Python interpreter. Slower than fork but portable and thread-safe.

**Speedup**
: Ratio of sequential to parallel execution time: S = T₁/Tₙ. Ideal (linear) speedup equals the processor count.

**Starvation**
: Condition where a process/thread cannot access resources it needs because others continuously acquire them first.

---

## T

**Task Graph**
: Directed acyclic graph representing computational dependencies. Nodes are tasks; edges are data dependencies. Dask builds and optimises task graphs before execution.

**Thread**
: Lightweight execution unit sharing memory with parent process. In Python, limited by GIL for CPU-bound work but effective for I/O.

**ThreadPoolExecutor**
: Executor using a pool of threads. Appropriate for I/O-bound tasks where GIL releases.

---

## W

**Worker**
: Process or thread in a pool that executes submitted tasks.

**Work Stealing**
: Load balancing technique where idle workers take tasks from busy workers' queues.

**Work-Span Model**
: Parallel analysis framework. Work (W) is total operations; Span (S) is critical path. Parallelism = W/S.

---

## Symbols and Formulae

| Symbol | Meaning |
|--------|---------|
| P | Parallelisable fraction of program |
| n | Number of processors/workers |
| S | Speedup |
| E | Efficiency |
| T₁ | Sequential execution time |
| Tₙ | Parallel execution time with n processors |
| W | Total work (operations) |
| S | Span (critical path length) |
| α | Serial fraction (Gustafson) |

**Key Formulae**:
- Amdahl's Law: S(n) = 1/((1-P) + P/n)
- Maximum speedup: S_max = 1/(1-P)
- Gustafson's Law: S(n) = n - α(n-1)
- Speedup: S = T₁/Tₙ
- Efficiency: E = S/n

---

*14UNIT — Parallel Computing and Scalability*
*Glossary v4.0.0*
