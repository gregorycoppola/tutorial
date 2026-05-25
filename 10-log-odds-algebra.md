# The Log-Odds Algebra: The Mathematical Spine

## Why This Algebra Matters

The connection between transformers and Bayesian networks is not just a structural observation — it runs through a specific algebra that appears independently in three places: cryptanalysis (Turing and Good, 1940s), probabilistic graphical models (Pearl, 1988), and the sigmoid transformer (Vaswani et al., 2017). The same formula, derived three times by people who did not know they were deriving the same thing.

Understanding this algebra is the key to understanding why the transformer IS a Bayesian network, not just looks like one.

## The Algebra

The logit function: logit(p) = log(p / (1-p)) maps (0,1) → R.
The sigmoid function: σ(x) = 1 / (1 + e^{-x}) maps R → (0,1).
These are exact inverses: σ(logit(p)) = p.

Under the logit isomorphism, the open interval (0,1) with the operation:

    m0 ⊕ m1 = σ(logit(m0) + logit(m1))

forms an abelian group with identity element 0.5.

Properties:
- Commutativity: m0 ⊕ m1 = m1 ⊕ m0
- Associativity: (m0 ⊕ m1) ⊕ m2 = m0 ⊕ (m1 ⊕ m2)
- Identity: m ⊕ 0.5 = m  (0.5 = maximum uncertainty, contributes nothing)
- Inverse: m ⊕ (1-m) = 0.5  (certainty and its opposite cancel to uncertainty)

## Turing and Good (1940s, Bletchley Park)

Context: breaking the Enigma cipher required combining independent pieces of evidence about the machine's daily settings. Each piece of evidence supported or weakened a hypothesis. The question: how do you combine N independent pieces of evidence correctly?

The answer: add their log-odds. Turing called this the "weight of evidence."

    W(H : e) = logit(P(H | e)) - logit(P(H))

For two independent pieces of evidence e0, e1:
    logit(P(H | e0, e1)) = logit(P(H)) + W(H : e0) + W(H : e1)

Starting from a uniform prior P(H) = 0.5, logit(P(H)) = 0, so:
    logit(P(H | e0, e1)) = logit(m0) + logit(m1)
    P(H | e0, e1) = σ(logit(m0) + logit(m1)) = m0 ⊕ m1

This is not an approximation. For truly independent evidence sources, it is the exact Bayesian update.

Good formalized this after the war (Good 1950, "Probability and the Weighing of Evidence"). The algebra is now called the "weight-of-evidence" framework.

## Pearl (1988)

Context: building a calculus for reasoning under uncertainty in AI systems. The belief propagation algorithm on binary variable nodes.

Pearl's sum-product update for a binary variable v with two independent incoming messages m0, m1:

    b_new(v) = m0 * m1 / (m0*m1 + (1-m0)*(1-m1))
             = σ(logit(m0) + logit(m1))

Pearl derived this from the general sum-product equations applied to binary variables with independent pairwise factors. He did not frame it as log-odds addition. He did not cite Turing and Good. The formula is the same.

On tree-structured factor graphs, this algorithm converges in diameter(T) rounds and produces exact marginal posteriors. Pearl proved this in 1988.

## The Sigmoid Transformer (2017)

Context: designing a sequence model for NLP. The FFN with sigmoid activation computes:

    output = σ(W2 · σ(W1 · x + b1) + b2)

With the BP weight construction (W1 extracting logit(m0) and logit(m1) from the residual stream, b1=0, W2=[1,1], b2=0):

    output = σ(logit(m0) + logit(m1)) = m0 ⊕ m1

The transformer FFN with BP weights IS the Turing-Good-Pearl formula. Not analogous to it. Not inspired by it. Computationally identical.

Vaswani et al. did not know they were implementing Turing and Good's weight-of-evidence combination. They were designing a neural network. The formula is the same.

## The Relation to Classical Boolean Logic

This algebra is the probabilistic generalization of classical Boolean AND and OR.

In the limit as beliefs approach 0 and 1:
- logit(1) = +∞  (certain true)
- logit(0) = -∞  (certain false)
- logit(0.5) = 0  (no information)

Two large positive numbers sum to a larger positive: high confidence in both inputs → high confidence in their conjunction. This is AND.

But equal and opposite evidence cancels: logit(p) + logit(1-p) = 0, giving 0.5. This differs from classical AND where FALSE AND TRUE = FALSE. The log-odds algebra gives maximum uncertainty instead — the correct behavior for uncertain reasoning.

Classical Boolean logic is the limiting case of this algebra as beliefs become certain.

## Why Sigmoid Is the Right Activation

The sigmoid function is not a design choice motivated by gradient flow or output normalization. It is the exact function required to implement this algebra.

A sigmoid FFN computing σ(w0·logit(m0) + w1·logit(m1) + b) is performing weighted log-odds addition and converting back to probability space in a single operation. This is the general Ψor factor from the QBBN — the OR gate with asymmetric weights and a prior bias.

Equal weights (w0=w1=1) and no bias (b=0) give the symmetric updateBelief. Unequal weights and nonzero bias give a skewed version — one evidence source matters more, or there is a prior favoring one outcome. This is the general factor potential.

## The Unification

Three independent derivations, three different contexts, same formula:

    Turing-Good (1943): combine independent cipher-breaking evidence
    Pearl (1988):       propagate beliefs in graphical models
    Sigmoid FFN (2017): transform residual stream in sequence models

The formula σ(logit(m0) + logit(m1)) is not a coincidence. It is the unique correct answer to the question: how do you combine two independent binary evidence sources into a posterior probability? Any other formula either violates the independence assumption, breaks commutativity, or produces outputs outside (0,1).

The transformer was always computing this. The algebra was always there. It took formal proof to name it.