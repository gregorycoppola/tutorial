# Component 3: The FFN as the OR / Update Step

## The BP Side

After the gather step assembles both neighbor beliefs into scratch slots, the update step computes a new belief:

    new_belief(j) = updateBelief(scratch[j][0], scratch[j][1])

where:

    updateBelief(m0, m1) = σ(logit(m0) + logit(m1))
                         = m0*m1 / (m0*m1 + (1-m0)*(1-m1))

This is the Bayesian update rule for two independent binary evidence sources. Log-odds add; sigmoid converts back to probability.

The algebra: logit(p) = log(p/(1-p)). Under this isomorphism, addition in R corresponds to multiplication of likelihood ratios — the correct operation for combining independent evidence. This is not an approximation. For truly independent evidence sources, it is exact.

Origin: developed by Alan Turing and I.J. Good at Bletchley Park for breaking Enigma, formalized by Good (1950), connected to graphical models by Pearl (1988). The formula is the same across all three; only the framing differs.

## The Transformer Side

The FFN applies independently to each token position:

    h = activation(W1 * x + b1)     — expand to D_ff dimensions
    output = W2 * h + b2             — project back to D dimensions

D_ff is large: 14,336 for 8B, 53,248 for 405B (≈ 3-4× D).

With sigmoid activation, the inner layer computes D_ff soft threshold operations. Each hidden unit fires if a particular combination of input dimensions crosses a threshold.

## The Identification

With BP weights and sigmoid activation:

    W1 extracts logit(m0) and logit(m1) from the specific dimensions where attention wrote them
    The hidden unit computes logit(m0) + logit(m1)
    W2 and sigmoid convert back: σ(logit(m0) + logit(m1)) = updateBelief(m0, m1)

The FFN is not computing a general nonlinear function. It is computing the Turing-Good-Pearl weight-of-evidence combination. The sigmoid activation is not a design choice for gradient flow. It is the exact function required to implement this algebra.

## The OR Contains an AND

Inside the OR computation there are two ANDs:

1. AND in the numerator: m0 * m1 is the joint probability both inputs are true (multiplication = AND for independent probabilities, or equivalently addition in log-odds space)
2. AND in the architecture: both m0 and m1 must be present in the residual stream before the FFN runs (enforced by the gather step)

The OR produces a probabilistic disjunction over causes: P(conclusion | evidence0, evidence1). The AND enforces the simultaneity of the evidence. The structure is OR(AND(e0, e1)).

## The Key-Value Memory Interpretation

Geva et al. (2021): FFN hidden units function as key-value memories.

- W1 rows = keys: patterns in the residual stream that trigger this unit
- W2 columns = values: what to write to the output if this unit fires

In BP terms: the key is the routing pattern (which combination of gathered evidence triggers this OR node). The value is the factor potential (what belief update this evidence produces). The FFN is a learned table of (pattern → belief update) pairs.

The number of distinct OR weight patterns = L × D_ff:
    Llama 3 405B: 126 × 53,248 = 6.7M distinct concept detectors

Each of the 6.7M hidden units is one conditional probability table entry in the implicit factor graph.

## ROME as Direct Evidence

Meng et al. (2022), ROME: Locating and Editing Factual Associations in GPT.

The key finding: factual associations ("The Eiffel Tower is in Paris") are stored in specific FFN weight matrices at specific layers. Editing those weights changes the factual association. The edit is local, precise, and transferable.

In Bayes net terms: ROME is directly editing factor potentials. The FFN weights ARE the conditional probability tables of the implicit factor graph. This is exactly what the BP interpretation predicts. ROME provides strong empirical evidence that the FFN-as-factor-potential identification is correct.

## SwiGLU and the Exactness Gap

The formal BP proof requires sigmoid activation. Production models (Llama, Mistral, etc.) use SwiGLU:

    SwiGLU(x) = SiLU(W1 * x) * (W3 * x)

SiLU(x) = x * sigmoid(x) ≈ ReLU(x) for x > 0. The gating multiplication introduces a second linear projection. This is not the exact BP activation.

However: the bayes-learner experiment (val MAE 0.000752) used a standard PyTorch TransformerEncoder with default ReLU activation and found near-exact BP weights. The qualitative OR structure holds because ReLU preserves non-negativity — the essential property for probabilistic interpretation. The exact proof requires sigmoid. The empirical finding says the structure is findable with compatible activations.

The gap between sigmoid (exact) and SwiGLU (compatible) is real but smaller than it looks. The key property is: outputs must be interpretable as unnormalized probability masses. ReLU, GELU, SiLU all satisfy this. Sigmoid handles normalization intrinsically; the others require an explicit normalization step (which softmax at the output provides).

## The OR/AND Ratio

D_ff / H = 416 for Llama 3 405B. There are 416 times as many OR patterns as AND patterns at every layer. The architecture is massively weighted toward inference (OR) over routing (AND). The model has a small fixed routing vocabulary and an enormous inference vocabulary. It knows many facts; it has few ways of deciding where to look.