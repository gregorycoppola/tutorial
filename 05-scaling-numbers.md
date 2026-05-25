# Component 5: The Scaling Numbers as a Bayes Net Inventory

## The Five Dimensions

A transformer has five architectural dimensions. Each has a precise meaning in BP terms.

    W  — sequence length      = nodes in the factor graph per forward pass
    D  — model dimension      = beliefs tracked per node
    H  — attention heads      = distinct neighbor lookups per node per round
    D_ff — FFN hidden dim     = OR nodes (concept detectors) per token per layer
    L  — layers               = rounds of BP per forward pass

These are not independent engineering choices. They are the parameters of the implicit Bayesian network.

## The Numbers for Key Models

Llama 3 8B:   L=32,  H=32,  D=4096,   D_ff=14336,  W=8192
Llama 3 70B:  L=80,  H=64,  D=8192,   D_ff=28672,  W=8192
Llama 3 405B: L=126, H=128, D=16384,  D_ff=53248,  W=8192

## The AND Inventory (Routing)

Distinct AND weight patterns = L × H (shared across all W positions).

    Llama 3 8B:   32 × 32  = 1,024
    Llama 3 70B:  80 × 64  = 5,120
    Llama 3 405B: 126 × 128 = 16,128

AND evaluations per forward pass = L × H × W.

    Llama 3 405B: 126 × 128 × 8,192 = 132M AND evaluations

16,128 distinct routing patterns for 405B is strikingly small. The model has a constrained routing vocabulary. It can ask only 16,128 distinct questions about "where should I look" — the same 16,128 questions applied at every one of the 8,192 token positions, at every one of the 126 layers.

## The OR Inventory (Inference)

Distinct OR weight patterns = L × D_ff (shared across all W positions).

    Llama 3 8B:   32 × 14,336  = 458,752 ≈ 460K
    Llama 3 70B:  80 × 28,672  = 2,293,760 ≈ 2.3M
    Llama 3 405B: 126 × 53,248 = 6,709,248 ≈ 6.7M

OR evaluations per forward pass = L × D_ff × W.

    Llama 3 405B: 126 × 53,248 × 8,192 = 55B OR evaluations

6.7M distinct concept detectors. Each is one conditional probability table entry in the implicit factor graph.

## The Ratio: OR Dominates AND

D_ff / H = 416 for Llama 3 405B.

There are 416 times as many OR patterns (concept detectors, inference) as AND patterns (routing, lookup) at every level. This ratio is consistent across model sizes:

    Llama 3 8B:   D_ff/H = 14336/32  = 448
    Llama 3 70B:  D_ff/H = 28672/64  = 448
    Llama 3 405B: D_ff/H = 53248/128 = 416

The architecture is consistently and heavily weighted toward inference over routing. The model has an enormous capacity to detect and combine evidence (OR) relative to its capacity to decide where to look (AND). In Bayesian network terms: the conditional probability tables are rich and numerous; the graph structure is sparse and constrained.

## What Scaling Adds

As models scale from 8B to 405B:
- L grows: 32 → 126 (more rounds of BP, deeper reasoning chains)
- H grows: 32 → 128 (more parallel lookups per round)
- D grows: 4096 → 16384 (more beliefs tracked per node)
- D_ff grows: 14336 → 53248 (more concept detectors per layer)
- W stays fixed at 8192 (same context length)

The factor graph gets richer (more beliefs per node, more concept detectors) and deeper (more rounds of inference). The number of nodes stays the same. Scaling is not adding more nodes — it is adding more rounds of richer inference over the same graph.

## The D/H Ratio: Head Dimension

Each head operates on D/H dimensions:
    Llama 3 8B:   4096/32  = 128 dimensions per head
    Llama 3 405B: 16384/128 = 128 dimensions per head

This ratio stays constant across scales. Each attention head always operates on 128-dimensional subspaces of the residual stream. Scaling adds more heads (more parallel lookups) but keeps each head's "view" the same size.

## The Total Proposition Count

Total proposition slots per forward pass = L × W × D.

    Llama 3 405B: 126 × 8,192 × 16,384 = 16.9B

Each of the 16.9B slots is one belief being tracked — one proposition about one token at one layer. This is the full state space of the implicit factor graph as it evolves across L rounds.

## Connecting to the Concept Space Theorem

Shannon, Theorem 9.2: for a factor graph with n nodes, the BP transformer operates over exactly 2n² distinct concepts (routing classes). The continuous parameters (belief values, factor table entries) are magnitude, not meaning.

For 405B with W=8192 positions: 2 × 8192² ≈ 134M distinct concept slots. But the actual routing vocabulary (L × H = 16,128) is the effective number of distinct conceptual roles the model recognizes. Most of the 134M slots are occupied by the same routing patterns applied to different token positions.

The 16K distinct AND patterns is the model's "conceptual alphabet" — the small set of distinct semantic roles it knows how to fill. The 6.7M OR patterns is its "inference vocabulary" — the large set of conditional relationships it has learned. Together they define the implicit factor graph.