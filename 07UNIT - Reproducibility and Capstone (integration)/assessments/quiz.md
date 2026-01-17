# Week 7 Quiz: Reproducibility and Testing

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Total Questions** | 10 |
| **Question Types** | 6 Multiple Choice, 4 Short Answer |
| **Time Limit** | 20 minutes |
| **Passing Score** | 70% |

---

## Multiple Choice Questions

### Question 1 (1 point)

Which of the following is **NOT** a common cause of the reproducibility crisis in computational research?

A) Undocumented software dependencies  
B) Using version control systems like Git  
C) Random seeds not being recorded  
D) Incomplete data preprocessing documentation

---

### Question 2 (1 point)

When setting random seeds for reproducibility in a Python project that uses both NumPy and the standard library, which approach is most robust?

A) Only set `random.seed()` since NumPy inherits from it  
B) Only set `np.random.seed()` since it's more comprehensive  
C) Set seeds for both `random` and `np.random` independently  
D) Use `os.environ['PYTHONHASHSEED']` which controls all randomness

---

### Question 3 (1 point)

In the pytest testing framework, what is the primary purpose of fixtures?

A) To measure code coverage  
B) To provide reusable test data and setup/teardown logic  
C) To generate random test inputs  
D) To format test output reports

---

### Question 4 (1 point)

Which testing strategy involves checking that code satisfies general properties rather than specific input-output pairs?

A) Unit testing  
B) Integration testing  
C) Property-based testing  
D) Regression testing

---

### Question 5 (1 point)

In a GitHub Actions workflow, what does the `matrix` strategy allow you to do?

A) Create branches automatically  
B) Run the same job multiple times with different configurations  
C) Merge pull requests automatically  
D) Generate code coverage reports

---

### Question 6 (1 point)

What is the primary purpose of a data manifest in reproducible research?

A) To compress data files for storage  
B) To record cryptographic hashes and metadata for verification  
C) To convert data between different formats  
D) To visualise data distributions

---

## Short Answer Questions

### Question 7 (2 points)

Explain the difference between **mocking** and **fixtures** in pytest, and provide one example use case for each.

*Write your answer in 3-5 sentences.*

---

### Question 8 (2 points)

A colleague claims their experiment is reproducible because they shared their Python script. What three additional pieces of information should they provide to ensure true reproducibility?

*List three items with brief explanations.*

---

### Question 9 (2 points)

Describe the purpose of the **AAA pattern** (Arrange-Act-Assert) in unit testing. Why is this pattern considered good practice?

*Write your answer in 2-4 sentences.*

---

### Question 10 (2 points)

What is the difference between **continuous integration (CI)** and **continuous deployment (CD)**? Give one example of what each automates in a research software project.

*Write your answer in 3-5 sentences.*

---

## Answer Key

<details>
<summary>Click to reveal answers</summary>

### Multiple Choice Answers

**Q1: B** — Using version control systems like Git

*Explanation*: Version control systems like Git are a **solution** to reproducibility issues, not a cause. They help track changes and enable others to access exact versions of code.

**Q2: C** — Set seeds for both `random` and `np.random` independently

*Explanation*: Python's `random` module and NumPy's random number generator use separate states. Setting one does not affect the other, so both must be seeded explicitly for full reproducibility.

**Q3: B** — To provide reusable test data and setup/teardown logic

*Explanation*: Fixtures in pytest provide a way to set up preconditions, provide test data and handle cleanup. They can be reused across multiple tests and support various scopes (function, class, module, session).

**Q4: C** — Property-based testing

*Explanation*: Property-based testing (as implemented by libraries like Hypothesis) generates random inputs and checks that general properties hold, rather than testing specific examples.

**Q5: B** — Run the same job multiple times with different configurations

*Explanation*: The matrix strategy allows you to test across multiple Python versions, operating systems or other variables by running the same workflow steps with different parameter combinations.

**Q6: B** — To record cryptographic hashes and metadata for verification

*Explanation*: Data manifests store hashes (e.g. SHA-256) of files along with metadata, allowing researchers to verify that data has not been modified or corrupted.

### Short Answer Rubric

**Q7** (2 points):
- **Fixtures** provide reusable setup code and test data. Example: Creating a temporary database connection for tests.
- **Mocking** replaces real objects with controlled substitutes. Example: Mocking an API call to avoid network requests during testing.
- Award 1 point for correct distinction, 1 point for valid examples.

**Q8** (2 points):
Three items from:
1. **Dependencies/environment**: List of packages with versions (requirements.txt or environment.yml)
2. **Random seeds**: All random seeds used in the experiment
3. **Data**: Access to original data or instructions to obtain it
4. **Hardware/software versions**: Python version, OS, relevant hardware specs
5. **Execution instructions**: Clear steps to reproduce the experiment

Award 0.5 points per valid item (max 1.5) plus 0.5 for quality of explanation.

**Q9** (2 points):
- **Arrange**: Set up test preconditions and inputs
- **Act**: Execute the code being tested
- **Assert**: Verify the expected outcomes
- Good practice because it makes tests readable, maintainable and clearly structured.
- Award 1 point for correct pattern description, 1 point for explaining benefits.

**Q10** (2 points):
- **CI**: Automatically runs tests and checks on every code change. Example: Running pytest and ruff on each pull request.
- **CD**: Automatically deploys code to production/users after tests pass. Example: Publishing a Python package to PyPI when a release tag is created.
- Award 1 point for correct distinction, 1 point for valid examples.

</details>

---

© 2025 Antonio Clim. All rights reserved.
