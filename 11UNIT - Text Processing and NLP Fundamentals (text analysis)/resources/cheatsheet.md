# Cheatsheet — 11UNIT

## Regex and parsing

- Prefer raw strings in Python: `r"\b\w+\b"`.
- Use named capturing groups for schema extraction: `(?P<user>...)`.
- Compile patterns once when applied repeatedly.
- Document assumptions about whitespace and delimiters.
- Anchor patterns when possible (`^...$`) to reduce accidental matches.

Regex is treated here as a research-oriented craft rather than a collection of library calls. The
central question is how an analyst moves from raw character sequences to defensible claims, given
that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Regex is treated here as a research-oriented craft rather than a collection of library calls. The
central question is how an analyst moves from raw character sequences to defensible claims, given
that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Regex is treated here as a research-oriented craft rather than a collection of library calls. The
central question is how an analyst moves from raw character sequences to defensible claims, given
that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Regex is treated here as a research-oriented craft rather than a collection of library calls. The
central question is how an analyst moves from raw character sequences to defensible claims, given
that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

## Tokenisation and statistics

- Define the tokeniser as a function `str -> list[str]` and test it.
- Measure vocabulary size as a function of corpus size.
- For TF–IDF: $\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\cdot \log\frac{N}{\mathrm{df}(t)+1}$.
- Track random seeds for any stochastic step.
- Use stratified splits for labelled corpora.

Tokenisation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Tokenisation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Tokenisation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Tokenisation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

## Evaluation

- Precision $= \frac{TP}{TP+FP}$, recall $= \frac{TP}{TP+FN}$, $F_1=\frac{2PR}{P+R}$.
- Inspect false positives and false negatives separately.
- Keep a frozen test set to avoid selection bias.
- Report confidence intervals where sampling variability is relevant.

Evaluation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Evaluation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Evaluation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Evaluation is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.
