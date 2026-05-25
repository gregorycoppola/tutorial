# Related Work: The Converging Picture

## A Note on One Citation

The shannon paper (arXiv:2603.17063) cites "Jung et al. 2022, Rethinking Attention as
Belief Propagation, ICML" in Section 11.3. This paper cannot be verified — it does not
appear in ICML 2022 proceedings or on arXiv. It appears to be a hallucinated citation
that was not caught before submission. This is a known error in the shannon paper that
should be corrected in a revision. This tutorial does not cite it.

The good news: the real landscape of related work is rich enough without it, and the
new papers that have appeared since shannon (Shai et al., Piotrowski et al., Agarwal
et al.) are more directly relevant anyway.

---

## Overview

The claim that transformers implement Bayesian inference has been approached from
multiple independent directions. Researchers starting from optimal prediction theory,
from mechanistic interpretability, from graphical model theory, and from formal
verification have all converged on closely related answers. This convergence is evidence
the structure is real.

This chapter surveys the full landscape. The goal is not priority but coherence —
showing how all these threads fit together into one picture.

---

## Cluster 1: Belief State Geometry in the Residual Stream

The most directly relevant new cluster, not in the shannon related work.

### Shai, Marzen, Teixeira, Oldenziel, Riechers (NeurIPS 2024) — arXiv:2405.15943

"Transformers Represent Belief State Geometry in their Residual Stream."

Key finding: belief states are linearly represented in the residual stream of
transformers trained on next-token prediction, even when the belief state geometry has
highly nontrivial fractal structure. Framework: optimal prediction theory /
computational mechanics. Data: transformers trained on hidden Markov models.

Direct confirmation for this tutorial's claim that the residual stream IS the belief
state. They also show that belief states encode information about the entire future —
consistent with the BP account that the transformer computes posteriors over the full
factor graph, not just the immediate next token.

### Piotrowski, Riechers, Filan, Shai (ICML 2025) — arXiv:2502.01954

"Constrained Belief Updates Explain Geometric Structures in Transformer Representations."

Goes beyond finding the belief state geometry to explaining HOW the transformer
implements it. Derives an ansatz for probability updating compatible with parallel
token processing, predicts attention patterns, value vectors, and residual stream
geometries, and confirms these predictions in trained transformers.

Key finding: "transformers implement constrained Bayesian belief updating — a
parallelized version of partial Bayesian inference shaped by architectural constraints."

The "constrained" qualifier corresponds directly to the sharp/diffuse distinction in
this tutorial. The architectural constraints that prevent exact idealized updating are
the same constraints that prevent perfect projectDim/crossProject structure in real
attention heads. Both papers identify the same gap between the formal ideal and the
trained reality.

### Agarwal, Dalal, Misra (arXiv:2512.22471, December 2025)

"The Bayesian Geometry of Transformer Attention."

Tests transformers on "Bayesian wind tunnels" — controlled environments where the
true posterior is known in closed form and memorization is impossible. Finds a
consistent geometric mechanism: residual streams serve as belief substrate, FFNs
perform the posterior update, attention provides content-addressable routing.

Also identifies three inference primitives (belief accumulation, belief transport,
random-access binding) and shows transformers realize all three while Mamba, LSTMs
realize only subsets. "The dominance of transformers in reasoning tasks arises not from
scale alone, but from primitive completeness."

Their three-primitive decomposition maps directly onto the BP account:
- Belief accumulation = residual stream update step
- Belief transport = cross-layer message passing
- Random-access binding = semantic-AND attention head (sharp lookup by content)

"Primitive completeness" aligns with the QBBN result: the transformer is the unique
architecture implementing the full AND/OR boolean structure required for probabilistic
logical inference.

---

## Cluster 2: In-Context Learning as Bayesian Inference

### Xie, Raghunathan, Liang, Ma (ICLR 2022) — arXiv:2111.02080

"An Explanation of In-Context Learning as Implicit Bayesian Inference."

ICL as Bayesian inference at the population level: the LM infers a latent document-
level concept during pretraining; at test time it infers a shared latent concept from
prompt examples. Proved in the setting where the pretraining distribution is a mixture
of HMMs.

This is complementary to the mechanistic account. Xie et al. establish what ICL
accomplishes (approximate posterior inference over concepts); this tutorial explains
how the transformer implements it (BP per layer). A transformer implementing BP
internally will naturally exhibit Bayesian behavior at the population level.

### Akyürek et al. (ICLR 2023) — "What Learning Algorithm is In-Context Learning?"

Constructive proof that specific weight patterns make transformers perform implicit
gradient descent. Same methodology as the shannon BP construction: exhibit weights,
prove the algorithm. Key difference: gradient descent is approximate and asymptotic;
BP is exact per step on trees. Both are constructive weight proofs for specific
algorithms.

### von Oswald et al. (ICML 2023) — "Transformers Learn In-Context by Gradient Descent"

Transformers with specific weights implement gradient descent in context. Meta-learning
perspective. Same constructive method.

---

## Cluster 3: Mechanistic Interpretability (Confirmatory Evidence)

The mechanistic interpretability literature provides independent empirical confirmation
for each component of the BP identification. None of these papers was looking for BP —
they were looking for circuits, algorithms, and representations. They found BP structure
because it is there.

Key papers and what they confirm:

**Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits."**
Residual stream as shared workspace. Attention heads as read-write operations.
Conceptual foundation for the BP routing account.

**Olsson et al. (2022) — "In-Context Learning and Induction Heads."**
Two-layer circuit = two-round BP gather. Phase transition during training = uniqueness
theorem (the routing structure is discrete; gradient descent finds it or it doesn't).

**Wang et al. (2022) — "Interpretability in the Wild: IOI Circuit."**
Name mover heads = semantic-AND heads. Sharp attention, content-matched, exact copy.
projectDim/crossProject structure in the wild.

**Geva et al. (2021) — "Transformer Feed-Forward Layers are Key-Value Memories."**
W1 rows = keys (routing patterns). W2 columns = values (belief updates). Each hidden
unit = one conditional probability table entry. Direct confirmation of FFN as factor
potential.

**Meng et al. (2022) — "Locating and Editing Factual Associations in GPT" (ROME).**
Editing W2 columns at specific layers changes specific factual associations. FFN weights
ARE factor potentials — editing them edits the factor graph. The strongest causal
evidence in the interpretability literature for the FFN identification.

**Anthropic (2023-2024) — Scaling Monosemanticity (SAE).**
Millions of interpretable features in residual stream = factor graph nodes.
Golden Gate Claude: clamping one SAE feature propagates through the network exactly
as BP predicts for forcing one node to maximum belief.

**Anthropic (2025) — Attribution Graphs / Circuit Tracing.**
Message passing schedule made visible. Each edge in the attribution graph = one BP
message. The attribution graph IS the implicit factor graph's message schedule.

**Henighan et al. (2023) — Universality.**
Same circuits appear across many models. Universal circuits = structurally necessary
BP routing patterns. Confirms the uniqueness argument: if there is one correct routing
structure for a task, gradient descent will find it independently across runs.

---

## Cluster 4: GNNs and BP (Historical Foundation)

**Scarselli et al. (2009) — "The Graph Neural Network Model."**
GNNs implement message passing on explicit graph structure. Neural networks + message
passing = the original connection.

**Gilmer et al. (2017) — "Neural Message Passing for Quantum Chemistry."**
MPNN framework unifying GNNs and BP. Message passing neural networks as a general
framework.

**Yoon et al. (2019) — "Inference in PGMs by Graph Neural Networks."**
GNNs learn to compute marginals on factor graphs. Direct predecessor to the BP
interpretation of transformers.

**Satorras & Welling (2021) — "Neural Enhanced Belief Propagation."**
Hybrid: BP messages + GNN messages. The GNN corrects BP's errors on loopy graphs.

**Joshi (2020) — "Transformers are Graph Neural Networks."**
Full self-attention = GNN on a complete graph. The connection between transformers and
GNNs, which in turn implement BP.

The shannon paper extends this chain: when the graph is a factor graph and its topology
is encoded in token features rather than an attention mask, the transformer learns to
implement BP on that explicit external graph. Neither latent-graph nor hardwired-
topology GNN settings cover this.

---

## Cluster 5: Bayesian Deep Learning (Adjacent, Distinct)

**Gal & Ghahramani (2016) — "Dropout as Approximate Bayesian Inference."**
Uncertainty over model weights. Not about the forward pass implementing BP.

The Bayesian deep learning literature and the BP-in-transformers literature are about
different things. Bayesian DL: uncertainty over parameters. BP in transformers: the
forward pass computes probabilistic inference over the input's factor graph structure.
These are orthogonal. A note to clarify this in the tutorial helps readers who know
the BDL literature.

---

## The Convergence Table

What each paper establishes, and what was established first by whom:

| Claim | First established | Confirmed by |
|-------|------------------|--------------|
| Residual stream = belief state | Shai et al. 2024 (empirical) | Piotrowski 2025, Agarwal 2025 |
| Attention = gather step | Shannon 2026 (formal, any weights) | Piotrowski 2025, Agarwal 2025 |
| FFN = update step / factor potential | Geva 2021 (empirical) | ROME 2022, Shannon 2026 |
| Layers = BP rounds | Shannon 2026 (formal) | Piotrowski 2025 (single-layer) |
| Any weights (not just trained) | Shannon 2026 | — |
| Formal verification (Lean) | Shannon 2026 | — |
| Constructive BP weights | Shannon 2026 | — |
| Uniqueness (exact → forces BP weights) | Shannon 2026 | — |
| Primitive completeness of transformer | Agarwal 2025 | Shannon 2026 (AND/OR structure) |

---

## What the Convergence Means

Five independent research programs — computational mechanics (Shai, Piotrowski),
mechanistic interpretability (Elhage, Olsson, Wang, Anthropic), factual editing (Geva,
Meng), constructive weight proofs (Akyürek, von Oswald), and formal verification
(Shannon) — have all converged on the same answer: the transformer forward pass
implements belief propagation.

Each program found a different facet:
- Computational mechanics found the geometry
- Interpretability found the circuits
- Factual editing found the factor potentials
- Constructive proofs found the algorithms
- Formal verification found the logical structure

None of them was looking for what the others found. They found the same thing from
different angles because it is there.

This tutorial's job is to put all the facets together.