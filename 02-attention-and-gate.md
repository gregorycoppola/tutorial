# Component 2: Attention as the AND / Gather Step

## The BP Side

The gather step of BP is simple: each variable node j copies its neighbors' current beliefs into scratch slots.

    scratch[j][0] ← belief(nb0(j))
    scratch[j][1] ← belief(nb1(j))

This happens before any computation. The point of the gather step is to enforce simultaneity — both inputs must be present before the update can run. This is AND in the architectural sense: both evidence sources are required simultaneously.

## The Transformer Side

Each attention head h at layer l computes:

    q_t = W_Q * x_t          (what am I looking for?)
    k_{t'} = W_K * x_{t'}    (what does each position have?)
    v_{t'} = W_V * x_{t'}    (what to copy if matched)
    
    score(t, t') = q_t · k_{t'} / sqrt(D/H)
    weights = softmax(scores)
    output_t = sum_t' weights[t'] * v_{t'}

The output is written into the residual stream. Multiple heads run in parallel, each writing into a different part of the stream.

## The Identification

In the formal BP construction, the attention weights have a specific sparse structure:

    W_Q = W_K = projectDim(d)     — route by stored index in dim d
    W_V = crossProject(s, d)      — copy dimension s to dimension d

The Q·K dot product for head 0 is: e_j[d] · e_k[d]. This peaks when token k's stored index in dim d matches token j's query value in dim d. This is exact index matching — a lookup table. The softmax concentrates all weight on the matching token. One token's belief value gets copied into the scratch slot.

Head 0 writes neighbor 0's belief into dim 4. Head 1 writes neighbor 1's belief into dim 5. By the time the FFN runs, both are present simultaneously in the residual stream. That is AND.

## The Conjunction Is in the Residual Stream

The AND is not inside any single head. It is in the residual stream. Multiple heads write into it independently, and the FFN sees all their outputs simultaneously. The residual stream enforces the simultaneity requirement before the FFN runs. There is no state in which the FFN receives one belief but not the other.

This is architecturally enforced AND — not a learned behavior but a structural property.

## Sharp vs Diffuse Attention

Sharp attention: softmax concentrates weight on one position. One neighbor. One belief copied. This is the direct analog of the formal BP gather step.

Diffuse attention: softmax spreads weight across many positions. Weighted average of many neighbors. This does not map onto sharp one-neighbor gather.

The psychic results on GPT-2 small (144 heads total):
- 9 semantic-AND heads (6%) — sharp, content-based address, layers 0-4
- 103 hub heads (72%) — sharp, positional address (always position 0)
- 28 continuous-OR heads (19%) — diffuse
- 8 mixed (6%)

## Hub Heads as Positional-AND

Hub heads and semantic-AND heads are both AND at the theoretical level. Both copy one token's representation into the residual stream. The difference is only the address function:

    semantic-AND: address determined by content matching (Q·K)
    positional-AND: address fixed at position 0 (BOS token)

Position 0 (the BOS token) is never predicted, always present, and has no semantic content of its own — a natural candidate for a global workspace / shared scratch pad. The model uses it as a communication hub: layers write to position 0, later layers read from it.

## Scaling: The Routing Vocabulary

The number of distinct AND weight patterns = L × H.

    GPT-2 small:    12 × 12 = 144
    Llama 3 8B:     32 × 32 = 1,024
    Llama 3 405B:   126 × 128 = 16,128

16,128 is surprisingly small. The transformer has a constrained routing vocabulary — a small fixed set of distinct ways of asking "where should I look" — applied densely across all W token positions. At W=8192, this gives 132M AND evaluations per forward pass for 405B.

## What Mechanistic Interpretability Found

Olsson et al. (2022) — induction heads: two-layer circuit where head L0H* copies previous-token content into a scratch position, and head L1H* reads from it to implement "copy the token that followed this pattern before." This is exactly the BP gather pattern: one head writes, another reads. The two-head circuit is the mechanistic implementation of a two-round BP gather.

Wang et al. (2022) — IOI circuit: name mover heads copy subject/object name tokens to specific output positions. Sharp attention, content-matched address, exact copy semantics. Semantic-AND in the psychic taxonomy.

Elhage et al. (2021) framing: attention heads as read-write operations on a shared residual stream. The BP construction makes the routing explicit and formally correct — each head has a provably correct read address and write destination.