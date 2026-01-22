# Week 10 Quiz: Data Persistence and Serialisation

## 📋 Quiz Information

| Property | Value |
|----------|-------|
| **Total Questions** | 10 |
| **Question Types** | 6 MCQ + 4 Short Answer |
| **Time Limit** | 20 minutes |
| **Passing Score** | 70% |

---

## Multiple Choice Questions

### Question 1 (1 point)

Which file mode should be used to write to a file without overwriting existing content?

- A) `'w'`
- B) `'r'`
- C) `'a'`
- D) `'x'`

---

### Question 2 (1 point)

What is the primary security concern when using Python's `pickle` module?

- A) Pickle files are not portable across operating systems
- B) Pickle can execute arbitrary code during deserialisation
- C) Pickle files are always larger than JSON
- D) Pickle cannot serialise custom classes

---

### Question 3 (1 point)

Which statement correctly opens a text file with explicit UTF-8 encoding?

- A) `open('file.txt', 'r')`
- B) `open('file.txt', 'r', encoding='utf-8')`
- C) `open('file.txt', 'rb', encoding='utf-8')`
- D) `open('file.txt', encoding='utf-8')`

---

### Question 4 (1 point)

In SQL, what does the ACID acronym stand for?

- A) Automated, Consistent, Isolated, Durable
- B) Atomicity, Consistency, Isolation, Durability
- C) Asynchronous, Concurrent, Independent, Distributed
- D) Automated, Cached, Indexed, Distributed

---

### Question 5 (1 point)

Which serialisation format provides the best human readability whilst supporting hierarchical data?

- A) CSV
- B) Pickle
- C) JSON
- D) Binary

---

### Question 6 (1 point)

What is the purpose of using parameterised queries in SQL?

- A) To improve query performance through caching
- B) To prevent SQL injection attacks
- C) To enable queries with multiple tables
- D) To compress query results

---

## Short Answer Questions

### Question 7 (2 points)

Explain why context managers (the `with` statement) are preferred for file operations in Python. Provide a specific scenario where not using a context manager could cause problems.

**Your Answer:**

```
[Write your answer here]
```

---

### Question 8 (2 points)

A research team needs to store 10 million temperature measurements from sensors. Each measurement includes a timestamp, sensor ID, and value. Compare the advantages of using CSV files versus an SQLite database for this use case. Which would you recommend and why?

**Your Answer:**

```
[Write your answer here]
```

---

### Question 9 (2 points)

Explain the concept of database normalisation. Give an example of how an unnormalised table could lead to data anomalies (update, insert, or delete anomalies).

**Your Answer:**

```
[Write your answer here]
```

---

### Question 10 (2 points)

A colleague suggests using SHA-256 checksums to verify data integrity after file transfers. Explain how this works and identify one limitation of this approach.

**Your Answer:**

```
[Write your answer here]
```

---

## Answer Key (Instructor Use Only)

<details>
<summary>Click to reveal answers</summary>

### Multiple Choice

1. **C) `'a'`** - Append mode adds to existing content without overwriting.

2. **B) Pickle can execute arbitrary code during deserialisation** - This makes it unsuitable for untrusted data.

3. **B) `open('file.txt', 'r', encoding='utf-8')`** - Mode and encoding must both be specified; 'rb' is for binary mode which doesn't use encoding.

4. **B) Atomicity, Consistency, Isolation, Durability** - The four properties that guarantee database transaction reliability.

5. **C) JSON** - Human-readable, supports nesting; CSV is flat, Pickle is binary.

6. **B) To prevent SQL injection attacks** - Parameterised queries separate code from data.

### Short Answer

7. **Context managers** ensure file handles are properly closed even when exceptions occur. Without them, if an exception is raised between `open()` and `close()`, the file handle leaks. Example: Writing to a file, encountering an error during processing, and the file remains locked/unclosed.

8. **CSV advantages**: Simple, universal, human-readable, easy to share. **SQLite advantages**: Indexed queries, ACID transactions, efficient filtering, no full-file loading for queries. **Recommendation**: SQLite for 10M records because querying specific time ranges or sensors would require loading entire CSV vs. millisecond index lookups in SQLite.

9. **Normalisation** systematically eliminates redundancy by organising data into related tables. **Example**: A table storing `(OrderID, CustomerName, CustomerAddress, Product)` repeats customer info for each order. **Update anomaly**: Changing a customer's address requires updating multiple rows. **Insert anomaly**: Cannot add a customer without an order. **Delete anomaly**: Deleting the last order loses customer info.

10. SHA-256 produces a unique fingerprint for file contents. After transfer, recomputing the checksum and comparing to the original verifies no bits were corrupted or modified. **Limitation**: It only detects changes, not which bytes changed or how to fix them. Also cannot verify data authenticity without a secure channel for the checksum itself.

</details>

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Week 10 — Quiz*

© 2025 Antonio Clim. All rights reserved.
