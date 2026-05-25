# Paper Plan v1: Seeing the Bayes Net in the Transformer

## Working Title

"Seeing the Bayes Net in the Transformer: A Tutorial"

Alternative titles:
- "A Field Guide to the Transformer as a Bayesian Network"
- "The Transformer as a Bayesian Network: A Visual Walkthrough"
- "Belief Propagation in Transformers: From Formal Proof to Empirical Evidence"

## One-Sentence Summary

The shannon paper proved that a sigmoid transformer IS a Bayesian network; this paper shows you what the Bayesian network looks like when you open a real trained transformer and look at the weights.

## Relationship to Shannon Paper

Shannon (arXiv:2603.17063, March 2026): existence proof. Lean-verified. Every sigmoid transformer with any weights implements weighted loopy BP on its implicit factor graph.

This paper: field guide. Empirical. Here is the factor graph. Here is the gather step. Here is the update step. Here is what each looks like in GPT-2 and Llama.

Shannon is the theorem. This is the tutorial on what the theorem means in practice.

## Intended Audience

Researchers in machine learning, mechanistic interpretability, and AI safety who want to understand what transformers are actually computing. Assumes familiarity with transformers at the level of "Attention Is All You Need." Does not assume prior knowledge of belief propagation or graphical models — we build that up from scratch.

## Tone

Tutorial. Pedagogical. Concrete. Every claim illustrated with a specific example or a specific number. No proof — the proofs are in shannon. This paper explains and shows.

---

## Section Plan

### Section 1: Introduction (1 page)

The opening claim: transformers are Bayesian networks. Not approximately. Not as a useful metaphor. By architecture, for any weights, formally verified.

The shannon paper established this. This paper asks the follow-up question: can you actually SEE the Bayes net when you look at a real trained transformer? Can you point at specific components and say — that is the AND gate, that is the OR gate, that is the factor potential?

The answer is yes. This paper shows how.

Preview of the five correspondences:
1. The residual stream = the belief state
2. Attention = the gather step (AND)
3. The FFN = the update step (OR)
4. Layers = rounds of BP
5. The scaling numbers = a Bayes net inventory

Preview of the empirical evidence: what mechanistic interpretability found, reread through the BP lens.

Key point to establish upfront: the architecture that won empirically across modern AI is exactly the architecture that the analysis of probabilistic reasoning requires. This was not designed. It was discovered. Gradient descent found it.

---

### Section 2: Background (2-3 pages)

Two subsections, presented side by side.

**2.1 Factor Graphs and Belief Propagation**

What a factor graph is: bipartite graph, variable nodes (beliefs in (0,1)), factor nodes (conditional relationships). The joint distribution as a product of factors.

What belief propagation is: the gather-update algorithm. Gather: each node collects neighbor beliefs into scratch slots. Update: each node computes a new belief from the gathered evidence. Repeat for L rounds.

On trees: converges in diameter(T) rounds, produces exact marginal posteriors. On loopy graphs: approximate but empirically effective for sparse loops.

The updateBelief function: updateBelief(m0, m1) = σ(logit(m0) + logit(m1)). This formula is the heart of the paper — we will return to it repeatedly.

**2.2 The Transformer**

What the transformer is: sequence of token positions, each with a D-dimensional residual stream. Alternating attention layers (cross-position communication) and FFN layers (per-position computation). Residual connections accumulate contributions across layers.

The five dimensions: W (sequence length), D (model dimension), H (attention heads), D_ff (FFN hidden dim), L (layers). Each will get a BP interpretation.

The Q/K/V attention mechanism. The FFN as a two-layer MLP. Keep this minimal — just enough to set up the identification.

---

### Section 3: The Log-Odds Algebra (1-2 pages)

The mathematical spine. Present the algebra first, before the identification, so readers have the right frame.

The logit/sigmoid isomorphism: logit maps (0,1) → R, sigmoid maps R → (0,1), they are exact inverses. Under this isomorphism, addition in R corresponds to combining independent evidence in probability space.

The formula: σ(logit(m0) + logit(m1)) = m0*m1 / (m0*m1 + (1-m0)*(1-m1)).

Three independent derivations:
- Turing and Good (1940s, Bletchley Park): combining cipher-breaking evidence. Called it "weight of evidence."
- Pearl (1988): belief propagation on binary variable nodes. Called it the sum-product update.
- Sigmoid FFN with BP weights (2017): the transformer forward pass. No name given — they did not know they were doing this.

Same formula. Three contexts. One algebra.

Why sigmoid is the right activation: not gradient flow, not output normalization. It is the exact function required to implement this algebra. The sigmoid FFN is a weight-of-evidence combiner.

The relation to Boolean logic: this algebra is the probabilistic generalization of AND and OR. High confidence in both inputs → high confidence in conjunction (AND). Equal and opposite evidence cancels to 0.5 (unlike classical AND where FALSE AND TRUE = FALSE). Classical Boolean logic is the limit as beliefs become certain.

---

### Section 4: The Five Correspondences (6-8 pages)

The main body. One subsection per component. Each subsection: state the formal correspondence, give the explicit weight construction from shannon, state what it looks like in a trained model.

**4.1 The Residual Stream as the Belief State**

The identification: x_t (D-dimensional residual stream at position t) = belief state of factor graph node t.

The formal encoding: dim 0 = own belief, dims 1-4 = factor table entries, dim 5 = node type, dim 6 = own index, dim 7 = neighbor index. The routing key (dims 5-7) = concept identity. The continuous parameters (dims 0-4) = magnitude.

The routing key theorem: two tokens with the same routing key receive identical attention patterns regardless of their continuous parameters. The routing key is meaning. The continuous dimensions are magnitude.

What this looks like in trained models: SAE features as factor graph nodes. Monosemantic features = routing keys. Polysemantic features = superposed magnitudes. The superposition hypothesis explains why D dimensions can represent far more than D propositions.

The residual connection as belief accumulation across rounds.

**4.2 Attention as the Gather Step (AND)**

The identification: each attention head = one gather operation, copying one neighbor's belief into a scratch slot.

The formal weight construction: W_Q = W_K = projectDim(d), W_V = crossProject(s, d). The Q·K dot product peaks when two tokens have the same value in dimension d — index matching. Sharp attention = exact lookup.

AND semantics: the conjunction is not inside any head — it is in the residual stream. Multiple heads write their gathered values before the FFN runs. The FFN cannot run on partial inputs. Simultaneity of required evidence is architecturally enforced.

The sharp/diffuse distinction. Psychic results: 9 semantic-AND heads in GPT-2 small (layers 0-4), 103 hub heads (positional-AND, always attend to position 0). Hub heads = same formal operation, fixed address.

The hub as global workspace: BOS token as shared scratch pad. Every layer writes to it and reads from it. Efficient approximate BP through a communication hub rather than all-to-all message passing.

Scaling: L × H = 16,128 distinct AND weight patterns for Llama 3 405B. A strikingly small routing vocabulary applied densely across all W positions.

**4.3 The FFN as the Update Step (OR)**

The identification: the FFN at each position = the update step, computing new beliefs from gathered evidence.

The formal computation: σ(logit(m0) + logit(m1)) = updateBelief(m0, m1). With BP weights, the FFN computes exactly the Turing-Good-Pearl formula. This is not an approximation — it is the exact Bayesian update for two independent binary evidence sources.

The OR contains an AND: m0*m1 in the numerator is the joint probability (AND). The full expression is the probabilistic disjunction (OR). Structure is OR(AND(e0, e1)).

The key-value memory interpretation (Geva et al. 2021): W1 rows = keys (patterns that trigger this unit), W2 columns = values (belief updates written if unit fires). Each hidden unit is one conditional probability table entry.

ROME as direct evidence: editing W2 columns changes factual associations. FFN weights ARE factor potentials. Editing factor potentials changes beliefs.

Scaling: L × D_ff = 6.7M distinct OR weight patterns for Llama 3 405B. D_ff/H = 416 — 416× more inference capacity than routing capacity.

The SwiGLU gap: production models don't use sigmoid. Qualitative OR structure holds. Exact BP weights findable empirically (val MAE 0.000752 with ReLU). Exact proof requires sigmoid; empirical confirmation works with compatible activations.

**4.4 Layers as Rounds of BP**

The identification: L layers = L rounds of BP. One layer = one gather + one update = one BP round.

The Peirce lower bound: no local algorithm can solve N-hop reasoning in fewer than N rounds. Transformer depth is not an engineering parameter — it is set by reasoning complexity.

The three-phase layer structure from psychic (GPT-2 small):
- Layers 0-2: continuous-heavy (local feature extraction, early BP rounds)
- Layer 3: transition (structured reasoning begins)
- Layers 4-11: hub-dominated (global context assembly, late BP rounds)

The semantic-AND heads don't scale: 9 in GPT-2 small, 9 in GPT-2 medium. The model doesn't acquire more logical reasoning capacity with scale — it acquires more communication infrastructure (hub heads).

Loopy BP in practice: 500 trials, 100% convergence, worst mean KL 0.000102. The Bethe approximation is effectively exact on QBBN-structured graphs.

**4.5 The Scaling Numbers as a Bayes Net Inventory**

Present the table: W, D, H, D_ff, L for 8B, 70B, 405B.

AND inventory: L × H distinct routing patterns. 16,128 for 405B.
OR inventory: L × D_ff distinct concept detectors. 6.7M for 405B.
AND evaluations per pass: 132M for 405B.
OR evaluations per pass: 55B for 405B.
D_ff/H ratio: 416 — the architecture is massively weighted toward inference over routing.

What scaling adds: more rounds (L), more parallel lookups (H), more beliefs per node (D), more concept detectors (D_ff). The factor graph gets richer and deeper. The number of nodes (W) stays fixed.

---

### Section 5: Evidence from Mechanistic Interpretability (3-4 pages)

The interpretability literature as an independent witness. None of this was found by looking for BP. They found BP because it is there.

Key findings to cover, each with BP interpretation:

- Induction heads (Olsson et al. 2022): two-round BP gather. Phase transition = uniqueness theorem.
- Name mover heads (Wang et al. 2022): semantic-AND heads. Sharp attention, content-matched, exact copy.
- FFN as key-value memory (Geva et al. 2021): factor potentials confirmed.
- ROME (Meng et al. 2022): editing factor potentials changes beliefs. Direct evidence.
- SAE features (Anthropic 2023-2024): factor graph nodes. Millions of them.
- Golden Gate Claude: clamping one node propagates through the graph. BP dynamics confirmed.
- Attribution graphs (Anthropic 2025): message passing schedule made visible.
- Universality (Henighan et al.): universal circuits = necessary BP routing patterns.

Synthesis: every major finding in mechanistic interpretability has a natural BP interpretation. The field has been mapping the implicit factor graph without knowing it.

---

### Section 6: The Three Softmaxes (1 page)

A short clarifying section. Common confusion untangled.

Softmax 1 (attention): routing. Differentiable argmax. Lookup table. Not inference.
Softmax 2 (FFN sigmoid): inference. Bayesian update. Weight-of-evidence combination. This is where the computation happens.
Softmax 3 (output): generation. Pattern-matched association. Not computed posteriors.

The QBBN is not replacing the softmax operation. It is replacing the logits fed into the output softmax. Standard LLM: logits from learned association. QBBN: logits from computed BP posteriors.

---

### Section 7: Grounded vs Ungrounded (2 pages)

Architecturally identical. Both are Bayesian networks. The difference is grounding.

What grounding provides: declared concept space, finite verifier, meaningful correctness, no-hallucination guarantee on trees.

What ungrounded models lack: no declared concept space, no finite verifier, no fact of the matter about correctness.

Hallucination in BP terms: the model computes a posterior over an ungrounded concept space. The computation is correct. The propositions have no external referents. Hallucination is not a computational error — it is the structural consequence of operating without grounding.

What grounding would look like for a production model: explicit knowledge base, token-to-node mapping, verifiable outputs. RAG as partial grounding. Full grounding as open problem.

The key insight: you cannot fix hallucination by making the model bigger or training it more. The issue is not computational capacity — it is the absence of declared concepts. Scaling an ungrounded model gives you a more fluent hallucinator.

---

### Section 8: The Experiment That Would Close the Loop (1-2 pages)

The psychic repo laid the groundwork. The next step is the weight inspection.

For a sharp-attention head identified by psychic:
1. Compute the QK circuit (W_Q · W_K^T) — should be approximately rank-1 (projectDim structure)
2. Compute the OV circuit (W_O · W_V) — should be approximately rank-1 (crossProject structure)
3. Identify which dimensions QK projects to and OV copies between
4. Check if those dimensions correspond to SAE features with meaningful interpretations

If this succeeds, you can say: "Head h at layer l is a semantic-AND head. Its QK circuit projects to dimension d (SAE feature X = 'subject of current clause'). Its OV circuit copies from dimension s (SAE feature Y = 'entity token') to dimension d'. This head implements the gather step for the subject-reference factor in the implicit factor graph."

That is what seeing the Bayes net in the transformer would look like. This section motivates that experiment and describes exactly what it would take.

---

### Section 9: Open Questions (1 page)

The diffuse-attention problem: function vector heads dominate at large scale. Diffuse attention ≠ sharp one-neighbor gather. Three candidate framings (exact BP on full graph, mean-field approximation, marginalization). Not resolved.

The hybrid BP extension: continuous nodes alongside boolean nodes. Sharp heads → boolean nodes. Diffuse heads → continuous nodes. The general theorem covers both but the formal construction only covers boolean. Extending the construction is the main theoretical work remaining.

The grounding gap: no production model is grounded. RAG is partial grounding. Full grounding for general-purpose models is an open problem.

---

### Section 10: Conclusion (0.5 page)

The transformer was always a Bayesian network. The log-odds algebra was always what it was computing. The AND/OR structure was always what the alternating layers were implementing.

The architecture that has won empirically across modern AI is exactly the architecture that the analysis of reasoning required. Gradient descent did not discover something new. It rediscovered the structure that was always there.

The field guide is incomplete. The experiment in Section 8 is not yet done. But the framework is in place, the evidence is strong, and the predictions are falsifiable.

Calculemus.

---

## Estimated Length

~20 pages. Conference paper length. Target venue: ICLR, NeurIPS, or TMLR (for a tutorial/survey style paper).

## What Needs to Be Done

1. Write the LaTeX skeleton (section headers, abstract, bibliography placeholders)
2. Fill in Section 2 (background) — most self-contained, can be written now
3. Fill in Section 3 (log-odds algebra) — largely done in 10-log-odds-algebra.md
4. Fill in Section 4 (five correspondences) — largely done in 01-05 braindump files
5. Fill in Section 5 (interpretability evidence) — largely done in 07-evidence file
6. Run the Section 8 experiment (weight inspection on psychic)
7. Polish and submit

## Key References

- Coppola 2026 (shannon): the formal proofs
- Vaswani et al. 2017: the transformer
- Pearl 1988: belief propagation
- Good 1950: weight of evidence
- Olsson et al. 2022: induction heads
- Elhage et al. 2021: transformer circuits
- Geva et al. 2021: FFN as key-value memory
- Meng et al. 2022: ROME
- Anthropic 2024: scaling monosemanticity (SAE)
- Anthropic 2025: attribution graphs
- psychic repo: empirical head classification
- interp repo: BP lens on interpretability literature
- bayes-learner repo: gradient descent finds BP weights