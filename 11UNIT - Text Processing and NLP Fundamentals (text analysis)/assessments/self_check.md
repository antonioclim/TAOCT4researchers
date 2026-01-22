# Self-check — 11UNIT

## Reflection prompts

- Which transformations in your pipeline are reversible, and what does irreversibility imply for later analysis?
- Where can you state invariants (for example, that token order is preserved) and how can you test them?
- Which errors are systematic and which are stochastic in your extractor?
- How would you justify a tokenisation choice to a reviewer concerned about construct validity?
- If your model fails on a new domain, which component would you examine first and why?

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.

Reflection is treated here as a research-oriented craft rather than a collection of library calls.
The central question is how an analyst moves from raw character sequences to defensible claims,
given that token boundaries, normalisation choices and annotation schemes impose inductive bias. We
therefore separate specification from implementation: we first state what a pipeline must preserve
or discard, then encode those constraints as pure functions with explicit preconditions. Attention
is paid to failure modes, including ambiguous segmentation, encoding artefacts and domain drift,
since these frequently dominate error budgets in empirical studies. In practice, the unit couples
formal definitions with executable checks so that each step has a measurable contract: input
alphabet, transformation invariants and complexity expectations.
