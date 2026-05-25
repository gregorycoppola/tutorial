# Component 4: The Layer Structure as Rounds of BP

## The BP Side

BP runs multiple rounds until convergence (on trees) or until a fixed number of rounds (on loopy graphs). Each round is one full gather + update pass over all nodes. The number of rounds needed equals the diameter of the factor graph — the longest shortest path between any two nodes.

On a tree of depth d, exactly d rounds are needed. More complex reasoning chains require more rounds.

## The Transformer Side

L transformer layers stack the same attention + FFN computation. L = 32 for 8B, 80 for 70B, 126 for 405B. Each layer is one round of BP. L layers = L rounds of BP.

The depth of the network is not an engineering hyperparameter chosen for capacity. It is determined by the complexity of the reasoning tasks the model needs to solve. A task requiring k hops of inference needs at least k layers.

## The Peirce Lower Bound

The pearl repo (in the logic mono) establishes a formal lower bound: no local algorithm can solve N-hop reasoning in fewer than N rounds. This is the Peirce lower bound on reasoning depth.

Transformers have this exact structure: L layers for L-hop reasoning. The scaling of depth with model capability is not arbitrary — it tracks the complexity of the tasks being learned.

## The Three-Phase Layer Structure (GPT-2 Small)

The psychic results reveal a clear three-phase structure:

Phase 1 — Early layers (0-2): continuous-heavy.
Dominated by continuous-OR heads doing real-valued aggregation. Boolean-AND heads sparse. Raw token embeddings get processed into useful representations. Local feature extraction. No structured cross-token reasoning yet.

Phase 2 — Transition (layer 3): mixed.
Boolean-AND, continuous-OR, hub, and mixed heads all appear. The network shifts from local feature extraction to global communication. The prev-token heads in this layer (3.2, 3.7) do BP-style routing at exactly the moment the network transitions to hub mode.

Phase 3 — Hub-dominated (layers 4-11): global workspace.
Nearly all heads communicate through position 0. Once early processing is done, the model's primary mechanism is reading from and writing to the shared communication channel at BOS. Information is assembled at position 0 and read out by later layers.

## What the Phase Structure Means in BP Terms

Phase 1 corresponds to the first rounds of BP: gathering local evidence, computing initial beliefs from raw inputs. On a factor graph, the first round propagates beliefs from observed evidence nodes to their immediate neighbors.

Phase 2 is the transition: beliefs from local evidence start reaching non-local nodes through multi-hop chains.

Phase 3 is the global workspace pattern: rather than passing messages directly between every pair of nodes (which would require W² attention), the model routes everything through a shared hub. This is efficient approximate BP — not exact (every node doesn't directly communicate with every other node) but effective.

The hub is position 0 (BOS). Every layer writes to it; every layer reads from it. It functions as the "current global belief state" — a summary of all evidence seen so far, updated each layer.

## The Semantic-AND Heads Don't Scale

A striking psychic finding: the number of semantic-AND heads (sharp, content-based routing) is constant across model sizes.

    GPT-2 small (144 heads):  9 semantic-AND heads
    GPT-2 medium (384 heads): 9 semantic-AND heads

The model does not acquire more logical reasoning capacity (semantic-AND) as it scales. It acquires more positional routing capacity (hub heads) — more global workspace infrastructure.

Interpretation: the 9 semantic-AND heads correspond to a fixed set of logical relations the model always needs (previous token, induction, name mover, etc.). These are established early and stay constant. What scales is the capacity to assemble and communicate context — more hub heads means more parallel reads from and writes to the global workspace.

## The Shannon Paper's Layer Statement

Shannon, Section 8.3: "A reasoning chain requiring k hops of inference requires k transformer layers. The depth of the network is not an engineering hyperparameter — it is determined by the depth of the reasoning chain in the factor graph."

This is a strong claim. It says that if you want to understand why transformers have the depths they do, you should look at the depth of the reasoning tasks they're trained on — not at optimization considerations like gradient flow or expressivity.

## Loopy BP and Production Models

The formal exactness guarantee requires tree structure. Most real knowledge bases are loopy. The loopy repo ran 500 trials across five graph structures and found: 100% convergence, worst mean KL divergence 0.000102. The Bethe approximation is effectively exact on QBBN-structured graphs.

Production transformers are doing loopy BP. The loops correspond to shared entities appearing in multiple rules, or conclusions feeding back as premises. In typical knowledge bases this is sparse. The resulting loops are long and the factor potentials are moderate — exactly the conditions under which loopy BP is known to converge.