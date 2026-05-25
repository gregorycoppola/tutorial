# Evidence from the Mechanistic Interpretability Literature

## The Program

Mechanistic interpretability reverse-engineers trained transformers to find interpretable circuits — subnetworks that implement specific algorithms. It is empirical, post-hoc, and model-specific.

The BP framework is formal, constructive, and general. It predicts what circuits should look like. Mechanistic interpretability goes and looks.

This file catalogs the key findings and what each means in BP terms.

## Induction Heads (Olsson et al. 2022)

Finding: a two-layer circuit where head L0H* copies previous-token content to a scratch position, and head L1H* reads from it to complete "copy the token that followed this pattern before."

BP interpretation: this is a two-round BP gather. Round 1: head L0H* implements gather step — copies neighbor belief (the token following the pattern) into a scratch position in the residual stream. Round 2: head L1H* reads from that scratch position — a second gather using the output of the first as its source.

The induction head circuit is the simplest possible multi-hop BP computation: two rounds, two heads, one message passed between them. The fact that this circuit emerges spontaneously during training is consistent with the uniqueness theorem — there is one correct routing structure for this task, and gradient descent finds it.

The phase transition in induction head formation (Olsson et al.: heads form suddenly at a specific training step) is consistent with the uniqueness result: the routing structure is discrete, not continuous. Gradient descent is either doing the right routing or not. The transition is when it snaps into place.

## Name Mover Heads (Wang et al. 2022 — IOI circuit)

Finding: in the indirect object identification task ("Mary and John went to the store. John gave a drink to ___"), specific heads copy the subject name token to the output position.

BP interpretation: name mover heads are semantic-AND heads. They implement a content-matched gather: Q/K matching identifies the token carrying the relevant name, V copies the name token's representation to the output. One neighbor, one lookup, exact copy. Exactly projectDim/crossProject.

The IOI circuit involves multiple heads working together: S-inhibition heads suppress the wrong name (negative message in BP terms), name mover heads copy the right name, backup heads provide redundancy (k-ary OR decomposition in BP terms).

## FFN as Key-Value Memory (Geva et al. 2021)

Finding: FFN hidden units function as key-value memories. The first layer (W1) is the key matrix; the second layer (W2) is the value matrix. Individual units fire on specific input patterns and contribute specific output vectors.

BP interpretation: this is the direct empirical confirmation of the FFN-as-factor-potential identification. The key is the routing pattern (which evidence combination triggers this unit). The value is the conditional probability (what belief this evidence produces). Each hidden unit is one row in a conditional probability table.

The finding that specific factual associations are stored in specific FFN layers (Geva et al.; confirmed by ROME) means the factor potentials are localized — a specific layer encodes a specific set of facts. This is consistent with the layer-as-round interpretation: earlier layers handle lower-level pattern matching, later layers handle higher-level factual associations.

## ROME — Locating and Editing Factual Associations (Meng et al. 2022)

Finding: factual associations are stored in FFN weight matrices at specific middle layers. Editing those matrices with a rank-1 update changes the association precisely and durably.

BP interpretation: ROME is editing factor potentials. The FFN weights at layer l are the conditional probability tables for the factor nodes at round l of BP. A rank-1 update to W2 at layer l changes one conditional relationship in the implicit factor graph.

This is strong empirical evidence for the FFN-as-factor-potential identification. If FFN weights were implementing arbitrary nonlinear functions, there would be no reason to expect that a rank-1 edit would change one specific factual association without disrupting others. The localization and specificity of ROME edits makes sense only if the FFN weights are encoding structured conditional relationships — factor potentials.

## SAE Features (Anthropic 2023-2024)

Finding: sparse autoencoders trained on residual stream activations find millions of interpretable features — monosemantic directions corresponding to specific concepts.

BP interpretation: SAE features are nodes in the implicit factor graph. Each feature is one proposition being tracked. The SAE is recovering the factor graph structure from the superposed residual stream.

The Golden Gate Claude experiment (Anthropic 2024): clamping a single SAE feature (the "Golden Gate Bridge" feature) to its maximum value causes the model to obsessively refer to the Golden Gate Bridge in all outputs. This is exactly what happens in BP when you force a node to maximum belief — the belief propagates through the graph and affects all related nodes.

The SAE gives a lower bound on the concept count: the number of SAE features is a lower bound on the number of factor graph nodes. For a 7B model, Anthropic found ~34M features. The concept count scales with model size.

## Attribution Graphs (Anthropic 2025)

Finding: attribution graphs trace how information flows from input tokens through the residual stream to output predictions, via specific attention heads and FFN layers.

BP interpretation: attribution graphs are factor graph edges made visible. Each edge in the attribution graph is a message passing step in the implicit BP computation. The attention heads correspond to gather steps (edges from variable nodes to factor nodes). The FFN layers correspond to update steps (edges from factor nodes back to variable nodes).

The attribution graph work is the closest the mechanistic interpretability community has come to directly observing BP in action. They are tracing the message passing schedule of the implicit factor graph.

## Universality (Henighan et al. 2023)

Finding: certain circuits (curve detectors, frequency detectors, multimodal neurons) appear across many different models trained on different data modalities. The same circuits emerge independently.

BP interpretation: universal circuits are the circuits that are necessary for BP on the most common factor graph structures. Sharp routing heads (previous token, induction, name mover) are universal because the factor graphs for language always require the same basic routing operations. The routing patterns are structural necessities, not learned quirks.

This is consistent with the uniqueness theorem: for a given factor graph structure, there is one correct routing pattern. If the same factor graph structure appears across many tasks, the same circuit will emerge across many models.

## What the Literature Confirms

The mechanistic interpretability literature, taken together, provides strong empirical confirmation for every component of the BP interpretation:

- Attention heads as gather steps: confirmed by induction heads, name movers, IOI circuit
- FFN as factor potentials: confirmed by key-value memories, ROME, factual localization
- Residual stream as belief state: confirmed by SAE features, attribution graphs
- Layer structure as BP rounds: confirmed by the three-phase structure, phase transitions
- Factor graph nodes: confirmed by SAE features, Golden Gate clamping experiment

None of this was found by looking for BP. The interpretability researchers were looking for circuits, algorithms, and representations. They found BP, because that is what is there.