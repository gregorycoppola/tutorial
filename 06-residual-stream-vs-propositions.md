# The Residual Stream Size vs Proposition Count Tension

## The Question

The formal BP construction uses a small model: D=8 dimensions per token, encoding exactly one belief, one factor table, one routing key. The residual stream is small and its structure is transparent.

A production transformer has D=16,384 dimensions per token. That is not 16,384 independent propositions being tracked simultaneously. What is the relationship between D and the number of propositions?

## The Superposition Hypothesis

Elhage et al. (2022), "Toy Models of Superposition": neural networks represent more features than they have dimensions by using nearly-orthogonal directions in high-dimensional space. A D-dimensional residual stream can represent O(D log D) or more features simultaneously, with each feature a direction rather than a basis vector.

This means D=16,384 is not a hard limit of 16,384 propositions. The actual number of represented features is much larger — potentially millions.

The Anthropic SAE work (Scaling Monosemanticity, 2024) trained sparse autoencoders on residual stream activations and found millions of interpretable features in a 7B model. The residual stream at any given layer represents far more concepts than its dimensional count suggests.

## The BP View of Superposition

In the formal BP construction, each proposition occupies its own dedicated dimension. There is no superposition — the encoding is sparse and each dimension has a single meaning.

In a trained LLM, propositions are superposed — many features share the same dimensions. This is the gap between the clean formal construction and the messy trained reality.

Why does superposition happen? The network has far more things to represent (millions of concepts) than dimensions to represent them in (16,384). It solves this by using interference-tolerant approximately-orthogonal directions. Two features can coexist in the same dimension as long as they are unlikely to be simultaneously active (sparse activation).

## The Effective Proposition Count

In the BP construction, the effective proposition count = W × D_eff, where D_eff is the number of independently representable propositions per token. With superposition, D_eff >> D.

SAEs give an empirical lower bound on D_eff. The Anthropic scaling study found ~34M features in a 7B model. For a 405B model, the count is likely in the hundreds of millions.

Compare to the OR inventory: 6.7M distinct concept detectors (L × D_ff). Each OR node fires on one specific feature combination. The OR inventory approximates the number of conditional probability table entries in the implicit factor graph.

The numbers are in the same ballpark — tens to hundreds of millions of represented concepts, tens of millions of learned conditional relationships between them. The model is tracking an enormous factor graph.

## The Routing Key vs the Superposed Feature

In the formal construction, the routing key (dims 5-7) and the continuous parameters (dims 0-4) are cleanly separated. The routing key determines which concept this token is; the continuous parameters hold the belief values.

In a trained model, this separation is approximate. The SAE work shows that some features are monosemantic (a single direction, stable meaning, like a routing key) and some are polysemantic (meaning varies with context, like superposed continuous parameters).

The monosemantic features correspond to the routing key — stable concept identities that the attention mechanism uses to decide where to look. The polysemantic features correspond to the continuous parameters — magnitudes that carry belief values but whose interpretation depends on context.

This is not a clean separation in trained models. But it is the right frame for understanding what is happening.

## The Dimensionality Tension

Your question: the FFN has dense weight matrices (W1 is D_ff × D, not sparse). In the formal BP construction, W1 should read only the specific dimensions where attention wrote the neighbor beliefs — a sparse, structured operation. A trained model has a dense W1 that reads all D dimensions.

This is the central tension between the formal construction and trained models. The formal construction says: W1 is sparse, reads dim 4 and dim 5 only. A trained model: W1 is dense, reads all 16,384 dimensions.

The resolution: in the trained model, the relevant information (the gathered neighbor beliefs) is mixed across many dimensions via superposition. W1 needs to be dense to extract it. The sparsity is not in the weight matrix — it is in the activation pattern. The hidden units fire sparsely (each unit activates for a small fraction of inputs), which is the trained analog of the constructed model's sparse routing.

So the formal construction is a special case where D is small enough that the relevant information can be placed in specific dedicated dimensions. The trained model generalizes this to high-dimensional superposed representations, where the same computation must be done via dense projections that effectively "unpack" the superposition.

## Summary

The residual stream has D dimensions. The model represents far more than D propositions via superposition. The OR inventory (6.7M for 405B) estimates how many conditional relationships are learned. The AND inventory (16K for 405B) estimates how many distinct routing patterns the model uses. These numbers describe the implicit factor graph — enormous, superposed, high-dimensional, but still a factor graph.