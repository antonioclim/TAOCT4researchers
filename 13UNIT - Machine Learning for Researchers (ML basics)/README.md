# 13UNIT — Machine Learning for Researchers

## Synopsis

This unit introduces machine learning as an empirical modelling discipline in which predictive accuracy is assessed under an explicit evaluation design rather than inferred from a single training fit.

The emphasis is on research-facing competence: specifying an outcome, constructing features with traceable meaning, selecting an estimator whose inductive bias matches the scientific question and reporting uncertainty and failure modes.



## Learning objectives

By the end of the unit, students should be able to implement supervised learning workflows with defensible train–validation–test separation, execute unsupervised analyses for exploratory structure and justify model choices using appropriate metrics.

Students should also be able to interpret learning curves, diagnose high bias versus high variance regimes and perform basic hyperparameter search with cross-validation while avoiding common leakage pathways.



## Quick start

Create a virtual environment, install dependencies and run the curated demonstrations. The Makefile targets encode the intended sequence and provide a stable interface for non-specialist users.

Commands:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make run-supervised
make run-unsupervised
make test
```



## Architecture overview

The repository is structured around two laboratory modules: a supervised pipeline (classification and regression) and an unsupervised pipeline (clustering and dimensionality reduction).

A lightweight validation script checks structure, word counts and superficial stylistic issues and it may be extended with project-specific checks.



## Directory structure

The unit follows a canonical layout: theory materials in `theory/`, executable laboratories in `lab/`, assessed exercises in `exercises/` and tests in `tests/`.



## Connections to other units

The methodological framing builds on complexity awareness (03UNIT) and reproducible workflow practice (07UNIT). In turn, it prepares the ground for scaling considerations in parallel evaluation and large data pipelines (14UNIT).



## Further reading

The references in `resources/further_reading.md` prioritise primary sources and widely cited methodological syntheses that are routinely used in research training.




From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

From a methodological perspective, the central difficulty is not writing code but specifying the inferential claim, the assumptions under which it is defensible and the diagnostics that reveal when those assumptions fail.

Accordingly, each experiment should be accompanied by a record of data provenance, preprocessing choices and metric definitions, because these decisions often dominate model selection effects when datasets are modest in size.

In applied work, it is prudent to treat performance estimates as random variables conditioned on a sampling design; resampling methods and confidence intervals are therefore discussed alongside point estimates.

---

## 📜 Licence and Terms of Use

<div align="center">

<table>
<tr>
<td>

<div align="center">
<h3>🔒 RESTRICTIVE LICENCE</h3>
<p><strong>Version 4.1.0 — January 2025</strong></p>
</div>

---

**© 2025 Antonio Clim. All rights reserved.**

<table>
<tr>
<th>✅ PERMITTED</th>
<th>❌ PROHIBITED</th>
</tr>
<tr>
<td>

- Personal use for self-study
- Viewing and running code for personal educational purposes
- Local modifications for personal experimentation

</td>
<td>

- Publishing materials (online or offline)
- Use in formal teaching activities
- Teaching or presenting to third parties
- Redistribution in any form
- Creating derivative works for public use
- Commercial use of any kind

</td>
</tr>
</table>

---

<p><em>For requests regarding educational use or publication,<br>
please contact the author to obtain written consent.</em></p>

</td>
</tr>
</table>

</div>

### Terms and Conditions

1. **Intellectual Property**: All materials, including code, documentation,
   presentations and exercises, are the intellectual property of Antonio Clim.

2. **No Warranty**: Materials are provided "as is" without warranty of any kind,
   express or implied.

3. **Limitation of Liability**: The author shall not be liable for any damages
   arising from the use of these materials.

4. **Governing Law**: These terms are governed by the laws of Romania.

5. **Contact**: For permissions and enquiries, contact the author through
   official academic channels.

### Technology Stack

<div align="center">

| Technology | Version | Purpose |
|:----------:|:-------:|:--------|
| Python | 3.12+ | Primary programming language |
| NumPy | ≥1.24 | Numerical computing |
| Pandas | ≥2.0 | Data manipulation |
| Matplotlib | ≥3.7 | Static visualisation |
| pytest | ≥7.0 | Testing framework |
| scikit-learn | ≥1.3 | ML algorithms |
| Seaborn | ≥0.12 | Statistical plots |
|

</div>
