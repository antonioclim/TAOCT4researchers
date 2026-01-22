# Unit 10: Assessment Rubric

## Homework Assignment Rubric

### Total Points: 100

---

## Part 1: Configuration Management System (20 points)

| Criterion | Excellent (18-20) | Good (14-17) | Satisfactory (10-13) | Needs Work (0-9) |
|-----------|-------------------|--------------|----------------------|------------------|
| **Functionality** | All functions work correctly with edge cases handled | Core functions work, minor edge cases missed | Basic functionality present, some errors | Functions incomplete or non-functional |
| **Code Quality** | Clean, well-documented, follows PEP 8 | Mostly clean with minor style issues | Readable but lacks documentation | Poor style, hard to read |
| **Error Handling** | Comprehensive try/except, meaningful messages | Good error handling, minor gaps | Basic error handling | Missing error handling |
| **Type Hints** | Complete and accurate type annotations | Most functions annotated | Some type hints present | Missing type hints |

---

## Part 2: Multi-Format Data Importer (25 points)

| Criterion | Excellent (23-25) | Good (18-22) | Satisfactory (13-17) | Needs Work (0-12) |
|-----------|-------------------|--------------|----------------------|------------------|
| **Format Detection** | Automatic detection works for all formats | Most formats detected correctly | Basic detection works | Detection fails or missing |
| **Data Transformation** | Consistent output structure, proper type conversion | Good transformation, minor inconsistencies | Basic transformation works | Transformation incomplete |
| **Streaming Support** | Memory-efficient streaming for large files | Streaming works but not optimised | Partial streaming implementation | No streaming support |
| **Validation** | Comprehensive validation with clear error messages | Good validation, minor gaps | Basic validation present | Validation missing |
| **Documentation** | Excellent docstrings, usage examples | Good documentation | Basic docstrings | Poor or missing documentation |

---

## Part 3: Research Database System (35 points)

| Criterion | Excellent (32-35) | Good (25-31) | Satisfactory (18-24) | Needs Work (0-17) |
|-----------|-------------------|--------------|----------------------|------------------|
| **Schema Design** | Normalised (3NF), appropriate indices, constraints | Well-structured, minor normalisation issues | Functional schema, some redundancy | Poor schema design |
| **CRUD Operations** | All operations work, parameterised queries | Core operations work correctly | Basic operations present | Operations incomplete |
| **Query Functions** | Complex queries efficient, proper JOINs | Good queries, minor inefficiencies | Basic queries work | Queries incorrect or missing |
| **Transactions** | Proper use of transactions, rollback on error | Transactions used, minor issues | Some transaction use | No transaction handling |
| **Data Integrity** | Foreign keys enforced, constraints validated | Good integrity checks | Basic constraints | Integrity not enforced |
| **Testing** | Comprehensive tests, edge cases covered | Good test coverage | Basic tests present | Missing tests |
| **Documentation** | Schema documented, clear function docstrings | Good documentation | Basic documentation | Poor documentation |

---

## Part 4: Data Integrity System (20 points)

| Criterion | Excellent (18-20) | Good (14-17) | Satisfactory (10-13) | Needs Work (0-9) |
|-----------|-------------------|--------------|----------------------|------------------|
| **Checksum Computation** | Efficient chunk-based, multiple algorithms | Works correctly, single algorithm | Basic implementation | Incorrect or missing |
| **Manifest Generation** | Complete metadata, proper structure | Good manifest format | Basic manifest works | Manifest incomplete |
| **Verification** | Detects all changes, clear reporting | Good detection, minor gaps | Basic verification works | Verification fails |
| **Performance** | Handles large files efficiently | Good performance | Acceptable performance | Poor performance |

---

## Code Quality Standards (Applied to All Parts)

### Style Guidelines
- Follow PEP 8 naming conventions
- Maximum line length: 88 characters
- Use meaningful variable names
- Consistent indentation (4 spaces)

### Documentation Requirements
- Module-level docstring explaining purpose
- Function docstrings with Args, Returns, Raises
- Inline comments for complex logic
- Type hints on all functions

### Testing Expectations
- Unit tests for each function
- Edge case coverage
- Clear test names indicating behaviour tested

---

## Submission Checklist

- [ ] All Python files run without syntax errors
- [ ] Type hints present on all functions
- [ ] Docstrings follow Google style
- [ ] No hardcoded paths (use pathlib)
- [ ] Context managers for all file operations
- [ ] Parameterised queries for all SQL
- [ ] UTF-8 encoding specified explicitly
- [ ] Tests pass successfully

---

## Late Submission Policy

| Days Late | Penalty |
|-----------|---------|
| 1 day | -10% |
| 2 days | -20% |
| 3 days | -30% |
| 4+ days | Not accepted |

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Unit 10 — Assessment Rubric*

© 2025 Antonio Clim. All rights reserved.
