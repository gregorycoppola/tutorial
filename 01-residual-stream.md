# Component 1: The Residual Stream as the Factor Graph Belief State

## The BP Side

In belief propagation, each variable node in the factor graph holds a belief — a scalar in (0,1) representing the current estimate of P(node = true). The belief gets updated each round as messages arrive from neighboring nodes. Between rounds, each node's belief is its "current best guess" given all evidence seen so far.

For a factor graph with N nodes, the full belief state at any round is a vector of N scalars, one per node.

## The Transformer Side

In the transformer, each token position t holds a D-dimensional residual stream vector x_t. This vector accumulates contributions from every layer — each attention head and FFN block adds to it via residual connections. By the final layer, x_t contains a rich representation of everything the model has computed about position t in context.

D is large: 4096 for an 8B model, 16384 for a 405B model. Each of those D dimensions is one "slot" in the representation.

## The Identification

The claim: each dimension of the residual stream is one proposition being believed. The residual stream vector at position t IS the belief state of the factor graph node corresponding to token t.

This is not just a metaphor. In the formal BP construction (shannon paper, Section 7), the token encoding assigns explicit meaning to each dimension:

    Dim 0:   own belief (initialized to 0.5)
    Dims 1-4: factor table entries [f00, f01, f10, f11]
    Dim 5:   node type (0 = variable, 1 = factor)
    Dim 6:   own index / (n-1)
    Dim 7:   neighbor index / (n-1)

Dims 5-7 are the routing key — the complete inferential identity of the token. Two tokens with the same routing key are the same concept, regardless of their continuous parameters. Dims 0-4 are the magnitudes — the actual belief values and factor strengths.

## The Routing Key as Concept Identity

This split is not a convenience of presentation. It is a theorem (shannon, Theorem 9.2): two tokens with the same routing key receive identical attention patterns regardless of their continuous parameters. The routing key is meaning. The continuous dimensions are magnitude.

For a factor graph with n nodes: |RoutingKey(n)| = 2 × n × n = 2n² distinct concepts. This is the implicit concept space of the BP transformer — finite, countable, grounded.

## What This Means for Trained Models

In a trained LLM, the residual stream dimensions don't have explicit labels. But the SAE (sparse autoencoder) work from Anthropic shows that individual directions in residual stream space correspond to interpretable features — "the Golden Gate Bridge", "the word 'and' in a mathematical context", etc. These are the implicit concept nodes of the ungrounded factor graph.

The routing key / magnitude split appears empirically as the monosemanticity / superposition phenomenon: some directions are dedicated to single concepts (monosemantic, like the routing key), others are polysemantic (like superposed magnitudes).

## The Residual Connection as Belief Accumulation

The residual connection x_{l+1} = x_l + Attn_l(x_l) + FFN_l(x_l) is exactly belief accumulation across rounds. Each round adds a delta — the result of one gather + one update — to the running belief state. The final residual stream is the belief state after L rounds of BP.

This is not a design choice motivated by gradient flow (though it helps with that too). It is the natural structure of iterative message passing: you start with a prior, add evidence at each round, and end with a posterior.