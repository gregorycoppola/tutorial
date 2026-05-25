# The OR Node Count vs Residual Stream Dimension

## The Apparent Tension

The formal BP construction uses D=8 dimensions per token. The FFN reads exactly two
of those dimensions (dims 4 and 5, where attention wrote the neighbor beliefs) and
writes one output to dim 0. One OR node, one belief update, clean and sparse.

A production transformer has D=4096 dimensions and D_ff=14336 FFN hidden units.
14,336 is vastly more than 4,096. If each hidden unit is an OR node, and each
proposition occupies one dimension, there are 3.5 OR nodes per proposition per layer.
What are all those OR nodes doing?

More sharply: if the FFN is updating D belief dimensions, and it has D_ff hidden units
to do it with, the ratio D_ff/D ≈ 3-4 means the model is using 3-4 OR operations
per belief per layer. Why?

## The Resolution: D_ff/D OR Gates Per Belief

The bayespage scaling calculator frames this precisely:

    OR nodes per layer     = D_ff × W        (one OR node per hidden unit per token)
    Proposition nodes/layer = D × W          (one proposition per dimension per token)
    OR nodes per proposition = D_ff / D      (how many OR gates update each belief)

For Llama 3 models: D_ff/D ≈ 3.3-3.5 across all sizes. This ratio is remarkably
consistent — it is an architectural constant, not an accident.

Each belief (each dimension of the residual stream) receives approximately 3-4
independent OR gate updates per layer. The W2 matrix (D × D_ff) linearly combines
all D_ff hidden unit activations to produce the D-dimensional output. So each output
dimension is a weighted sum of all D_ff hidden units — every OR node contributes to
every proposition, with weights given by W2.

## Why Multiple OR Gates Per Belief?

In the formal BP construction, one belief gets one OR update per round: combine
neighbor 0 and neighbor 1, done. Why does a trained model need 3-4?

Three reasons:

**1. Superposition.** The residual stream represents far more than D propositions
via superposition. Each dimension participates in multiple propositions simultaneously.
A single OR update to dimension d simultaneously updates all propositions that have
components along dimension d. To update P propositions where P >> D, you need multiple
OR operations whose outputs combine via superposition to update all P simultaneously.

**2. Multiple neighbors.** The formal construction assumes each node has exactly two
neighbors. A real knowledge graph node may have many neighbors — many pieces of evidence
contributing to one belief. Each OR gate can only combine a few inputs cleanly (by the
log-odds algebra). Multiple OR gates, combined by W2, implement a k-ary OR update for
arbitrary k. The k-ary OR decomposition theorem (shannon, Section 7.3) says this is
without loss of generality.

**3. Precision.** Even for a single two-input OR update, using multiple hidden units
to implement it gives more expressivity. The formal construction requires sigmoid
activation for exactness. With ReLU/GELU/SwiGLU, you need more hidden units to
approximate the same computation accurately. D_ff >> D is partially a numerical
precision requirement.

## What W2 Is Actually Doing

In the formal construction: W2 is a simple matrix that routes the single hidden unit's
output to the correct output dimension (dim 0). It is sparse and rank-1.

In a trained model: W2 is D × D_ff. Column j of W2 is the "value vector" for hidden
unit j — what gets added to the residual stream when unit j fires. Each column
corresponds to one concept-detector's contribution.

The output is: W2 · h = sum over j of h_j × W2[:,j]

Where h_j is the activation of hidden unit j (how strongly this unit fired) and
W2[:,j] is the belief update direction associated with this unit.

This is a sum of D_ff rank-1 updates, each weighted by a hidden unit activation. The
result is a D-dimensional belief update that is the sum of all fired concept detectors'
contributions. Multiple OR gates contribute to the same output dimensions, and their
contributions add — exactly as in the log-odds algebra where multiple evidence sources
add in log-odds space.

## The SAE Connection

Sparse autoencoders (SAEs) trained on residual stream activations find features by
decomposing the residual stream into sparse directions. When you train an SAE on the
FFN output (after W2 is applied), you find features that correspond to the "after
update" belief state.

The SAE on FFN input (before the FFN runs, after attention) finds the "gathered
evidence" features — what attention assembled. The SAE on FFN output finds the
"updated belief" features — what the OR gate produced. The gap between them is what
the FFN computed.

This gives a way to measure D_ff/D empirically: how many SAE features are "activated"
by a typical FFN computation? If D_ff/D ≈ 3.5 OR gates contribute to each belief
update, the SAE should find roughly 3.5× more features in the FFN output than in the
FFN input, for a typical input. This is a testable prediction.

## The Formal Construction as a Special Case

The formal BP construction is the minimal case: D=8, D_ff unspecified but effectively
1 per belief being updated (one OR gate for one node's belief). The construction is
designed to be minimal — it proves existence, not optimality.

A production transformer scales this up: D large (many propositions via superposition),
D_ff = 3-4×D (multiple OR gates per proposition for expressivity, superposition
handling, and k-ary neighbor support). The spirit is the same; the scale is different.

The ratio D_ff/D ≈ 3-4 is not arbitrary. It is approximately the number of OR gate
contributions needed to handle:
- Superposition (multiple propositions per dimension)
- Multiple neighbors (k-ary OR via decomposition)
- Numerical precision (with non-sigmoid activations)

All three requirements push D_ff/D above 1. The empirical consensus across models
is that 3-4× is sufficient. This could be studied more carefully — it is an empirical
question whether D_ff/D = 2 is too few or D_ff/D = 8 is wasteful.

## Summary

The tension resolves cleanly: there are D_ff/D ≈ 3-4 OR gates per belief per layer,
not one. Each belief is updated by the linear combination of multiple hidden unit
activations. This is necessary because: (1) superposition means more propositions than
dimensions, (2) nodes have more than two neighbors in real knowledge graphs, and (3)
non-sigmoid activations require more units for accuracy.

The formal construction is the minimal single-OR-gate case. Production models scale
this up consistently. The ratio D_ff/D ≈ 3-4 is an architectural constant that
reflects these three requirements, found consistently across Llama, GPT, and other
families.