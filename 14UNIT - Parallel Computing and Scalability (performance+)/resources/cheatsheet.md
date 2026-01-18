# 14UNIT: Quick Reference Cheatsheet

## Parallel Computing and Scalability

*Print-friendly: 2 A4 pages*

---

## Decision Tree: Which Approach?

```
Is your task CPU-bound or I/O-bound?
│
├─► CPU-bound (computation-heavy)
│   │
│   ├─► Data fits in memory?
│   │   ├─► Yes → multiprocessing.Pool
│   │   └─► No → Dask
│   │
│   └─► Need shared state?
│       ├─► Yes → Value/Array + Lock
│       └─► No → Pool.map (preferred)
│
└─► I/O-bound (network, disk, database)
    │
    └─► ThreadPoolExecutor or asyncio
```

---

## multiprocessing Essentials

### Process Creation
```python
from multiprocessing import Process, Pool, Queue, Value, Lock

# Single process
p = Process(target=func, args=(arg1,))
p.start()
p.join()  # Wait for completion

# Process pool
with Pool(processes=4) as pool:
    results = pool.map(func, items)        # Preserves order
    results = pool.starmap(func, args_list) # Multiple args
    async_r = pool.apply_async(func, args)  # Non-blocking
```

### Inter-Process Communication
```python
# Queue (FIFO, process-safe)
q = Queue()
q.put(item)
item = q.get(timeout=5)

# Shared Value
counter = Value('i', 0)  # 'i' = integer, 'd' = double
with counter.get_lock():
    counter.value += 1

# Shared Array
arr = Array('d', 100)  # 100 doubles
```

### Synchronisation
```python
lock = Lock()

# Option 1: Context manager
with lock:
    critical_section()

# Option 2: Explicit
lock.acquire()
try:
    critical_section()
finally:
    lock.release()
```

---

## concurrent.futures Essentials

```python
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)

# Process pool (CPU-bound)
with ProcessPoolExecutor(max_workers=4) as exe:
    results = list(exe.map(func, items))

# Thread pool (I/O-bound)
with ThreadPoolExecutor(max_workers=10) as exe:
    futures = [exe.submit(func, arg) for arg in args]
    
    # Process as completed (any order)
    for f in as_completed(futures):
        result = f.result()
    
    # With timeout
    result = future.result(timeout=5.0)
    
    # Check for exception
    exc = future.exception()
```

---

## Dask Essentials

### Delayed Computation
```python
from dask import delayed, compute

@delayed
def process(x):
    return expensive_op(x)

# Build graph (lazy)
results = [process(item) for item in items]
total = delayed(sum)(results)

# Execute
final = total.compute()

# Or compute multiple
a, b, c = compute(x, y, z)
```

### Dask Arrays
```python
import dask.array as da

# From NumPy (specify chunks)
arr = da.from_array(np_array, chunks=(1000, 1000))

# Random array
arr = da.random.random((10000, 10000), chunks='auto')

# Operations (lazy)
result = (arr ** 2).mean()

# Execute
value = result.compute()
```

### Dask DataFrames
```python
import dask.dataframe as dd

# Read CSV (automatic partitioning)
ddf = dd.read_csv('data_*.csv')

# Operations (lazy)
result = ddf.groupby('col').agg({'value': 'sum'})

# Execute
df = result.compute()

# Persist to memory (for reuse)
ddf = ddf.persist()
```

### Scheduler Selection
```python
# Global setting
import dask
dask.config.set(scheduler='processes')

# Per-computation
result.compute(scheduler='threads')

# Options: 'synchronous', 'threads', 'processes', 'distributed'
```

---

## Profiling Essentials

### cProfile
```python
import cProfile
import pstats

# Profile function
cProfile.run('my_function()', 'output.prof')

# Analyse
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10

# Or inline
profiler = cProfile.Profile()
profiler.enable()
result = my_function()
profiler.disable()
profiler.print_stats()
```

### Memory Profiling
```python
import tracemalloc

tracemalloc.start()
result = memory_intensive_function()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

### Timing
```python
import time

start = time.perf_counter()
result = function()
elapsed = time.perf_counter() - start
```

---

## Key Formulae

| Formula | Expression | Use |
|---------|------------|-----|
| Amdahl's Law | S = 1/((1-P) + P/n) | Speedup limit |
| Max Speedup | S_max = 1/(1-P) | Theoretical maximum |
| Speedup | S = T₁/Tₙ | Actual improvement |
| Efficiency | E = S/n | Resource utilisation |
| Monte Carlo SE | σ/√n | Sampling precision |

**Quick calculations**:
- 90% parallel (P=0.9) → max 10× speedup
- 95% parallel (P=0.95) → max 20× speedup
- 99% parallel (P=0.99) → max 100× speedup

---

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| No threading speedup | GIL | Use multiprocessing |
| `PicklingError` | Lambda/closure | Use module-level function |
| `BrokenProcessPool` | Worker crash | Check for exceptions |
| Race condition | Unsync'd shared state | Add Lock |
| Deadlock | Lock ordering | Consistent order, timeouts |
| Memory explosion | Data copied per process | Shared memory or Dask |
| Slow startup | Process overhead | Reuse pools |

---

## Type Hints Reference

```python
from multiprocessing import Pool, Queue
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, TypeVar, Sequence

T = TypeVar('T')
R = TypeVar('R')

def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    n_workers: int = 4
) -> list[R]:
    with Pool(n_workers) as pool:
        return pool.map(func, items)
```

---

## Import Quick Reference

```python
# Standard library
from multiprocessing import (
    Process, Pool, Queue, Pipe,
    Value, Array, Lock, Semaphore,
    Event, Barrier, Manager, cpu_count
)
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor,
    Future, as_completed, wait
)
import threading
import cProfile
import pstats
import tracemalloc

# Third-party
import dask
import dask.array as da
import dask.dataframe as dd
from dask import delayed, compute
from dask.diagnostics import ProgressBar
```

---

## Quick Tips

- Always use `if __name__ == '__main__':` guard for multiprocessing code
- Prefer `ProcessPoolExecutor` context manager over manual Pool management
- Set `chunksize` parameter for large iterables to reduce communication overhead
- Use `multiprocessing.freeze_support()` for Windows executables

---

*14UNIT — Parallel Computing and Scalability*
*Quick Reference v4.0.0*
