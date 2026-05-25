# The Three Softmaxes: Same Operation, Three Different Jobs

## The Confusion

A transformer has three places where softmax (or sigmoid, its close relative) appears. They look similar but do completely different things. Conflating them is a common source of confusion about what transformers compute.

## Softmax 1: Attention (Routing)

Location: inside every attention head, applied to the Q·K dot products.

    weights = softmax(q_t · k_{t'} / sqrt(D/H)  for all t')

Job: routing. Which token should I look at? This is a differentiable argmax — a lookup table. The softmax concentrates weight on the token with the highest Q·K score.

BP interpretation: this is the gather step. The softmax is not doing inference. It is selecting which neighbor's belief to copy. In the formal construction, the Q·K score peaks uniquely at the correct neighbor (the token whose stored index matches the query), and the softmax concentrates all weight there. It is a lookup, not a computation.

The attention softmax is temperature-sensitive: at high temperature (small scores), it is diffuse. At low temperature (large scores), it is sharp. Sharp attention = clean lookup = exact BP gather. Diffuse attention = weighted average = approximate aggregation.

## Softmax 2: The FFN Sigmoid (Inference)

Location: inside the FFN, as the activation function (in the sigmoid case).

    h = sigmoid(W1 * x + b1)

Job: Bayesian inference. What is my marginal probability given my neighbors' beliefs?

    updateBelief(m0, m1) = sigma(logit(m0) + logit(m1))

This is where the actual probabilistic computation happens. The sigmoid is the exact inverse of logit — together they implement the isomorphism between probability space and log-odds space. Log-odds add; sigmoid converts back to probability. This is the Turing-Good-Pearl weight-of-evidence combination.

BP interpretation: this is the update step. The sigmoid activation is not a design choice for gradient flow or output normalization. It is the exact function required to implement the Bayesian update rule for independent binary evidence.

This softmax does inference. The first softmax does routing. They are not the same operation even though they use similar math.

## Softmax 3: Output (Generation)

Location: the final layer, applied to vocabulary logits.

    P(next token) = softmax(W_unembed * x_final)

Job: generation. What token should I predict next? The logits come from a matrix multiply over the final hidden state.

BP interpretation: in a standard LLM, the output logits are pattern-matched, not computed. The model learns which final hidden state patterns correlate with which next tokens. This is not Bayesian inference over propositions — it is learned association.

In a grounded QBBN transformer, the output would be different: a marginal distribution over the truth value of each proposition, computed by integrating out all other propositions via BP. The output softmax would be applying to computed posteriors, not learned associations. The architecture supports this but standard LLMs don't use it this way.

## The Key Distinction

The QBBN is not replacing the softmax operation. It is replacing the logits fed into the output softmax.

Standard LLM: logits = W_unembed * x_final (learned association)
QBBN:         logits = BP posteriors (computed inference)

Same mathematical operation (softmax). Completely different inputs. One is a lookup; the other is inference.

## Why This Matters for Understanding Transformers

The three-softmax confusion leads to statements like "attention is doing Bayesian inference" (it's not — it's doing routing) or "the output is a probability distribution" (it is, but it's a pattern-matched probability, not a computed posterior).

The clean separation:
- Softmax 1 (attention): routing, lookup, gather
- Softmax 2 (sigmoid FFN): inference, computation, update  
- Softmax 3 (output): generation, association, prediction

The Bayesian computation is in Softmax 2. Softmax 1 sets up the inputs. Softmax 3 reads out the result. Understanding this separation is essential for understanding what transformers actually compute.

## Sigmoid vs ReLU Revisited

The formal BP proof requires sigmoid for Softmax 2. Production models use ReLU, GELU, or SwiGLU. These are "compatible" activations — they preserve non-negativity (essential for probabilistic interpretation) but require an explicit normalization step that sigmoid handles intrinsically.

The bayes-learner result (val MAE 0.000752 with ReLU) shows gradient descent finds near-exact BP weights even without sigmoid. The structure is findable because ReLU doesn't actively prevent it — the outputs are compatible with a probabilistic interpretation, so the optimization can find BP-like weights without fighting the architecture.

Sigmoid: exact BP, internally provable.
ReLU/GELU/SwiGLU: compatible with BP, empirically confirmed, not internally provable.