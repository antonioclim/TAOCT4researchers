# Lecture notes — 13UNIT: Machine Learning for Researchers

## From statistical modelling to learning systems

In many research settings, the distinction between statistical modelling and machine learning is less a difference in mathematics than a difference in workflow commitments: machine learning requires explicit out-of-sample evaluation and disciplined separation between model selection and final reporting.

We treat a dataset as a finite sample from an underlying data-generating process. Any performance estimate is therefore conditional on the sampling design and the evaluation protocol.



## Generalisation, risk and empirical estimation

Let $(X, Y)$ denote a random pair with distribution $P$. A predictor $f$ incurs risk $R(f)=\mathbb{E}[\ell(f(X),Y)]$ for a loss function $\ell$.

In practice we estimate $R$ by an empirical risk under a split or resampling design. Cross-validation approximates the expected performance over resampled training sets, whereas a held-out test set approximates performance on future draws when the split is representative.



## Bias–variance and learning curves

For squared error, the expected generalisation error decomposes into bias, variance and irreducible noise. The decomposition is informative even when the loss is not squared, because the qualitative diagnostics carry across.

Learning curves measure performance as a function of training set size. If both training and validation errors are large and similar, the model is underfitting. If the gap is large, the model has high variance and may benefit from regularisation or more data.



## Supervised learning workflows

Classification and regression pipelines differ primarily by the loss and evaluation metrics. The interface for data preprocessing should be identical: deterministic transformations, fitted on training data only, then applied to validation and test data.

In scikit-learn, `Pipeline` and `ColumnTransformer` formalise this discipline. Their chief benefit is not convenience but the elimination of leakage through a single fitted object.



## Unsupervised learning workflows

Clustering is not a substitute for a causal explanation. Nevertheless, it is useful for exploratory grouping, anomaly detection and dimensionality reduction when used with appropriate scepticism.

Because there is often no ground-truth label, evaluation relies on internal criteria (silhouette score, Davies–Bouldin index) and external validation when labels exist for auxiliary tasks.



## Hyperparameter tuning and nested validation

Hyperparameter optimisation must not reuse the test set. When model selection is extensive, nested cross-validation provides an unbiased performance estimate at a higher computational cost.

A practical compromise is to maintain a strict test set and tune on cross-validated performance within the training set.



## Reporting and reproducibility

A research-facing report should specify the evaluation protocol, the number of repeats or folds, the random seeds, metric definitions and the full preprocessing pipeline.

Where appropriate, report confidence intervals or variability across folds and state the plausible threats to validity, such as covariate shift or label noise.



## Common failure modes

Data leakage: preprocessing fitted on the full dataset, target leakage through engineered features, or temporal leakage when the split ignores time ordering.

Metric mismatch: optimising accuracy when class imbalance demands precision–recall based measures, or reporting $R^2$ without a baseline in small samples.




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