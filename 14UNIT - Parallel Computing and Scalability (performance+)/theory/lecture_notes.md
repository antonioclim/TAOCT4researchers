# 14UNIT: Lecture Notes

## Parallel Computing and Scalability

---

## Introduction

Modern computational research confronts an uncomfortable reality: datasets grow faster than processor speeds. Moore's Law, which predicted exponential growth in transistor density, has slowed; clock speeds have plateaued near 5 GHz for over a decade. The path forward lies not in faster individual processors but in parallel execution across multiple cores, processors and machines.

This unit develops practical skills for parallel computing in Python, addressing the language's unique architectural constraints whilst providing effective strategies for both CPU-bound and I/O-bound workloads. The treatment progresses from multiprocessing fundamentals through concurrent.futures abstractions to the Dask framework for out-of-core computation.

---

## §1. Fundamentals of Parallelism

### 1.1 Parallelism versus Concurrency

These terms, often conflated in casual discourse, describe fundamentally different phenomena with distinct implications for program design and performance.

**Parallelism** involves the simultaneous execution of multiple computations on separate processing units. True parallelism requires multiple CPU cores—or multiple machines—executing instructions at the same physical instant. A four-core processor can genuinely execute four independent computations simultaneously.

**Concurrency** involves managing multiple tasks that may overlap in time but need not execute simultaneously. A single core can achieve concurrency through interleaved execution—switching between tasks rapidly enough to create the illusion of simultaneity. A web server handling hundreds of simultaneous connections on a single core exemplifies concurrent (but not parallel) execution.

The distinction matters practically. A web server benefits from concurrency because most time is spent waiting for network I/O; the CPU sits idle while packets traverse the network. A Monte Carlo simulation benefits from parallelism because computation dominates; the CPU executes instructions continuously.

Python's Global Interpreter Lock complicates this picture, as we shall see.

### 1.2 Processes versus Threads

Operating systems provide two fundamental abstractions for concurrent execution:

**Processes** are independent execution environments with separate memory spaces. Communication between processes requires explicit mechanisms—pipes, queues, shared memory segments. Process creation carries significant overhead (fork/spawn system calls, memory allocation), but isolation provides reliability: one process crashing does not affect others.

**Threads** are lightweight execution units sharing memory with their parent process. Communication between threads is trivial—they share the same address space—but this shared state creates synchronisation challenges. Race conditions, deadlocks and data corruption lurk wherever multiple threads access shared data without proper coordination.

The choice between processes and threads depends on workload characteristics:

| Characteristic | Prefer Processes | Prefer Threads |
|----------------|------------------|----------------|
| Memory sharing | Explicit is acceptable | Essential for performance |
| Fault isolation | Critical | Acceptable to share fate |
| Startup cost | Amortised over long tasks | Frequent task creation |
| Python GIL | CPU-bound work | I/O-bound work |

### 1.3 The Global Interpreter Lock

CPython—the reference Python implementation—employs a Global Interpreter Lock (GIL) that ensures only one thread executes Python bytecode at any moment. This design choice simplifies memory management: reference counting, Python's primary garbage collection mechanism, requires no synchronisation when only one thread modifies object references.

The GIL's implications for parallel computing are profound:

```python
# This achieves NO speedup for CPU-bound work
import threading

def compute_sum(n):
    return sum(range(n))

# Two threads share one GIL—effectively sequential
threads = [
    threading.Thread(target=compute_sum, args=(10_000_000,))
    for _ in range(4)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Despite four threads, the GIL serialises execution. Total time approximates four sequential calls, not one quarter.

The GIL releases in specific circumstances:
- During I/O operations (file, network, database)
- During certain C extension operations (NumPy, SciPy)
- Explicitly via `Py_BEGIN_ALLOW_THREADS` in C extensions

For pure Python CPU-bound code, multiprocessing provides the escape route: separate interpreter processes, each with its own GIL, execute genuinely in parallel.

---

## §2. Amdahl's Law and Speedup Limits

### 2.1 Theoretical Foundation

In 1967, Gene Amdahl formalised the theoretical limits of parallel speedup. His observation, now known as Amdahl's Law, states that if a fraction *P* of a program can be parallelised and *(1-P)* must remain sequential, the maximum speedup with *n* processors is:

$$S(n) = \frac{1}{(1-P) + \frac{P}{n}}$$

As *n* approaches infinity, the speedup asymptotically approaches:

$$S_{max} = \frac{1}{1-P}$$

This limit proves sobering. If 90% of a program parallelises (*P* = 0.9), the maximum achievable speedup is 10×—regardless of processor count. The sequential 10% imposes an insurmountable bottleneck.

### 2.2 Practical Implications

Consider a data processing pipeline:
1. Load data from disk (5% of time, sequential)
2. Parse and validate (10% of time, parallelisable)
3. Compute transformations (70% of time, parallelisable)
4. Aggregate results (5% of time, partially parallelisable)
5. Write output (10% of time, sequential)

With 15% inherently sequential (*P* = 0.85), maximum speedup is approximately 6.7×. Adding more than eight processors yields diminishing returns.

### 2.3 Gustafson's Counterpoint

John Gustafson's 1988 reframing observes that researchers rarely fix problem size and add processors. Instead, they fix time budget and solve larger problems:

$$S(n) = n - \alpha(n-1)$$

where α represents the serial fraction. This "scaled speedup" perspective proves more relevant for scientific computing, where ambitions expand to match available resources.

---

## §3. Multiprocessing in Python

### 3.1 The multiprocessing Module

Python's multiprocessing module provides process-based parallelism that bypasses the GIL:

```python
from multiprocessing import Process, Pool, Queue

def worker(x):
    return x * x

# Single process
p = Process(target=worker, args=(5,))
p.start()
p.join()

# Process pool
with Pool(processes=4) as pool:
    results = pool.map(worker, range(100))
```

### 3.2 Pool Patterns

The Pool abstraction manages a collection of worker processes, distributing tasks and collecting results:

**map**: Apply function to each element, return results in input order
```python
results = pool.map(func, iterable)
```

**starmap**: Like map, but unpacks argument tuples
```python
results = pool.starmap(func, [(a1, b1), (a2, b2)])
```

**apply_async**: Non-blocking submission with optional callback
```python
async_result = pool.apply_async(func, args, callback=handle_result)
result = async_result.get(timeout=10)
```

### 3.3 Inter-Process Communication

Processes cannot share memory directly; explicit communication mechanisms are required:

**Queue**: Thread/process-safe FIFO queue for passing objects
```python
q = Queue()
q.put(item)
item = q.get(timeout=5)
```

**Pipe**: Two-way communication channel between two processes
```python
parent_conn, child_conn = Pipe()
parent_conn.send(data)
data = child_conn.recv()
```

**Value/Array**: Shared memory for primitive types
```python
counter = Value('i', 0)  # Shared integer
with counter.get_lock():
    counter.value += 1
```

---

## §4. Synchronisation and Coordination

### 4.1 Race Conditions

Race conditions occur when program behaviour depends on the relative timing of concurrent operations. Consider:

```python
# UNSAFE: Race condition
counter = Value('i', 0)

def increment():
    for _ in range(100000):
        counter.value += 1  # Read-modify-write is not atomic
```

Multiple processes incrementing simultaneously may read the same value before either writes, losing increments.

### 4.2 Locks and Mutual Exclusion

Locks ensure only one process accesses a resource at a time:

```python
from multiprocessing import Lock

lock = Lock()

def safe_increment():
    for _ in range(100000):
        with lock:  # Acquire/release automatically
            counter.value += 1
```

### 4.3 Additional Primitives

**Semaphore**: Counting lock permitting *n* simultaneous holders
```python
sem = Semaphore(3)  # Allow 3 concurrent accesses
```

**Event**: Simple signalling mechanism
```python
event = Event()
event.set()    # Signal
event.wait()   # Block until signalled
```

**Barrier**: Synchronise *n* processes at a common point
```python
barrier = Barrier(4)  # Wait for all 4 to arrive
barrier.wait()
```

---

## §5. Threading for I/O-Bound Work

### 5.1 When Threading Helps

Despite the GIL, threading provides genuine benefit for I/O-bound workloads. The GIL releases during I/O operations, allowing other threads to execute:

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_url(url):
    response = requests.get(url)  # GIL released during network wait
    return response.text

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_url, urls))
```

### 5.2 concurrent.futures Abstraction

The concurrent.futures module provides a high-level interface for parallel execution:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(func, arg) for arg in args]
    
    # Process results as they complete
    for future in as_completed(futures):
        try:
            result = future.result()
        except Exception as e:
            handle_error(e)
```

---

## §6. Dask: Scalable Parallel Computing

### 6.1 Motivation

When datasets exceed memory or computations span clusters, lower-level primitives become unwieldy. Dask provides:

- Familiar NumPy/pandas interfaces for larger-than-memory data
- Lazy evaluation enabling optimisation before execution
- Multiple schedulers from single-machine to distributed clusters

### 6.2 Dask Delayed

The `@delayed` decorator marks functions for lazy evaluation:

```python
from dask import delayed

@delayed
def process(x):
    return expensive_computation(x)

@delayed
def combine(results):
    return aggregate(results)

# Build graph (no computation yet)
parts = [process(chunk) for chunk in data]
result = combine(parts)

# Execute
final = result.compute(scheduler='processes')
```

### 6.3 Dask Arrays and DataFrames

Dask provides parallel versions of NumPy arrays and pandas DataFrames:

```python
import dask.array as da
import dask.dataframe as dd

# Dask array: chunked for parallel processing
arr = da.from_array(large_array, chunks=(1000, 1000))
mean = arr.mean().compute()

# Dask DataFrame: partitioned across files
ddf = dd.read_csv('data_*.csv')
result = ddf.groupby('category').mean().compute()
```

### 6.4 Scheduler Selection

Dask supports multiple execution backends:

| Scheduler | Use Case | Trade-offs |
|-----------|----------|------------|
| `synchronous` | Debugging | No parallelism, easy stack traces |
| `threads` | NumPy-heavy, I/O | GIL limits pure Python |
| `processes` | CPU-bound Python | Process overhead, no shared memory |
| `distributed` | Clusters | Network overhead, complex setup |

---

## §7. Performance Profiling

### 7.1 The Necessity of Measurement

Donald Knuth's observation applies especially to parallel computing: "Premature optimisation is the root of all evil." Intuition about parallel performance frequently misleads; only measurement reveals truth.

### 7.2 cProfile for Function-Level Analysis

```python
import cProfile
import pstats

cProfile.run('function_to_profile()', 'output.prof')

stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### 7.3 line_profiler for Line-Level Analysis

```python
# Install: pip install line_profiler
# Decorate function with @profile
# Run: kernprof -l script.py
# View: python -m line_profiler script.py.lprof
```

### 7.4 Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive():
    large_list = [i ** 2 for i in range(10_000_000)]
    return sum(large_list)
```

---

## §8. Monte Carlo Methods: A Parallel Framework

### 8.1 Embarrassingly Parallel Computation

Monte Carlo methods exemplify "embarrassingly parallel" computation—problems decomposing into independent subproblems requiring no inter-process communication. The textbook observes: "At the intuitive core of Monte Carlo methods lies a counterintuitive proposition: we can channel randomness to compute deterministic quantities."

### 8.2 Parallel π Estimation

The classic dartboard method parallelises trivially:

```python
def worker_pi(args):
    n_points, seed = args
    rng = random.Random(seed)
    inside = sum(
        1 for _ in range(n_points)
        if rng.random()**2 + rng.random()**2 <= 1
    )
    return inside

with Pool(n_workers) as pool:
    counts = pool.map(worker_pi, [(points_per_worker, i) for i in range(n_workers)])
    
pi_estimate = 4 * sum(counts) / total_points
```

### 8.3 Convergence and Standard Error

Monte Carlo standard error decreases as σ/√n—doubling accuracy requires quadrupling samples. This relationship makes parallelism particularly valuable: linear speedup in sample generation translates directly to faster convergence.

---

## §9. Recommended Approaches

### 9.1 General Guidelines

1. **Profile before parallelising**: Identify actual bottlenecks
2. **Prefer coarse-grained parallelism**: Minimise communication overhead
3. **Use appropriate abstractions**: concurrent.futures before raw threading
4. **Test sequential first**: Easier debugging, correctness verification
5. **Measure speedup empirically**: Compare against Amdahl's predictions

### 9.2 Common Pitfalls

| Pitfall | Cause | Solution |
|---------|-------|----------|
| No speedup | GIL limiting threads | Use multiprocessing |
| Pickle errors | Lambda/closure serialisation | Module-level functions |
| Memory explosion | Data copied per process | Shared memory or Dask |
| Deadlock | Circular lock dependencies | Consistent ordering |

### 9.3 When Not to Parallelise

Parallelisation adds complexity. Avoid it when:
- Sequential performance suffices
- Tasks are too small (overhead dominates)
- Strong data dependencies prevent decomposition
- Debugging parallel code would consume more time than saved

---

## §10. Advanced Considerations

### 10.1 Process Start Methods

Python's multiprocessing module supports three start methods with distinct characteristics:

**spawn** (default on Windows and macOS 3.8+): Creates a fresh Python interpreter process. The parent process pickles necessary data and sends it to the child. This method is portable across platforms but incurs higher startup overhead. Note that spawn requires that the main module be importable—code that executes at import time will run in child processes.

**fork** (default on Unix): Creates a child process by duplicating the parent's memory space via the operating system's fork() system call. Fast startup, but potentially unsafe when combined with threading—locks may be copied in locked state, causing deadlocks. Fork is not available on Windows.

**forkserver**: A compromise approach where a server process is started early, then used to fork workers on demand. Safer than raw fork when threads may be present.

```python
import multiprocessing as mp

# Set globally (before any process creation)
mp.set_start_method('spawn')

# Or use context for specific Pool
ctx = mp.get_context('spawn')
with ctx.Pool(4) as pool:
    results = pool.map(func, items)
```

### 10.2 Pickle Limitations

Data transfer between processes requires serialisation. Python's pickle module handles most built-in types but imposes constraints:

- Lambda functions and local functions cannot be pickled
- File handles, network connections, and locks cannot be pickled
- Large objects incur serialisation overhead

For large NumPy arrays, consider using shared memory to avoid copying:

```python
from multiprocessing import shared_memory

# Create shared memory block
shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
np.copyto(shared_array, array)

# Workers access same memory block by name
shm_worker = shared_memory.SharedMemory(name=shm.name)
```

### 10.3 Process Pool Sizing

Determining optimal pool size requires consideration of several factors:

For CPU-bound work, `cpu_count()` provides a reasonable starting point, though memory bandwidth and cache effects may favour slightly fewer workers. Hyperthreading typically provides limited benefit for compute-intensive tasks—physical cores matter more than logical cores.

For I/O-bound work with threading, higher worker counts often help since threads spend most time waiting. Network latency and server-side rate limits typically constrain practical parallelism before thread count becomes limiting.

Empirical measurement remains essential. A simple scaling study—measuring throughput with 1, 2, 4, 8 workers—often reveals the point of diminishing returns for specific workloads.

### 10.4 Error Propagation

Exceptions in worker processes require explicit handling. With `Pool.map()`, the first exception terminates processing and propagates to the caller. With `apply_async()`, exceptions are captured in the `AsyncResult` object:

```python
async_result = pool.apply_async(potentially_failing_function, args)

try:
    result = async_result.get(timeout=30)
except Exception as e:
    logger.error(f"Worker failed: {e}")
```

For production systems, consider wrapping worker functions to capture and return exceptions explicitly, enabling partial results even when some tasks fail.

---

## Summary

Parallel computing transforms computational research capabilities but demands new skills and careful analysis. The GIL constrains Python threading; multiprocessing provides the escape route for CPU-bound work. Amdahl's Law governs theoretical speedup limits; empirical profiling reveals practical performance. Dask extends familiar interfaces to larger-than-memory datasets. Monte Carlo methods exemplify embarrassingly parallel computation. Measurement, not intuition, guides effective optimisation.

The techniques developed in this unit apply throughout computational research: simulation studies requiring many iterations, machine learning at scale, large-scale data analysis and research computing infrastructure. Mastery of parallel computing distinguishes researchers who push practical boundaries from those constrained by single-processor limitations.

The journey from sequential to parallel thinking represents a fundamental shift in computational reasoning. Where sequential algorithms proceed step-by-step through well-defined states, parallel algorithms must reason about concurrent state, synchronisation points, and the subtle interactions between independent computations. This shift, once internalised, enables researchers to identify parallelisation opportunities throughout their work—from embarrassingly parallel simulation ensembles to carefully orchestrated data pipelines.

---

## References

- Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*, 30, 483-485.
- Beazley, D. (2010). Understanding the Python GIL. *PyCon 2010*.
- Gorelick, M., & Ozsvald, I. (2020). *High Performance Python* (2nd ed.). O'Reilly Media.
- Gustafson, J. L. (1988). Reevaluating Amdahl's law. *Communications of the ACM*, 31(5), 532-533.
- Herlihy, M., & Shavit, N. (2012). *The Art of Multiprocessor Programming*. Morgan Kaufmann.

---

*14UNIT — Parallel Computing and Scalability*
*Lecture Notes v4.0.0*
