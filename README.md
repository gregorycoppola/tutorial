# Finding the Bayes Net in the Transformer

A tutorial paper on message passing, belief propagation, and transformer circuits.

**Greg Coppola, 2026**

## What this is

This is the source for the tutorial companion to the paper
[Transformers are Bayesian Networks](https://arxiv.org/abs/2603.17063).
The companion paper proves — formally, in Lean 4 — that a sigmoid transformer
can exactly realize belief propagation under a constructive weight mapping.
This tutorial explains what that result means, walks through the transformer
architecture component by component, and connects the formal result to a
converging body of empirical work in mechanistic interpretability, in-context
learning, and graphical model theory.

The compiled PDF is `index.pdf`.

## The core claim

A sigmoid transformer does not merely resemble belief propagation at a high
level. Under the constructive mapping, the transformer forward pass implements
the BP update equations directly:

- The residual stream is the belief state
- Attention performs evidence gathering (the AND gate)
- The FFN performs belief update (the OR gate)
- Layers are rounds of message passing
- The scaling dimensions describe inference capacity

## Structure

The paper is organized in five parts:

- **Part I — The Mathematical Foundation**: the log-odds algebra of Turing,
  Good, and Pearl; the constructive weight mapping; Turing completeness
- **Part II — The Five Correspondences**: residual stream, attention, FFN,
  layers, scaling numbers
- **Part III — Going Deeper**: superposition and the Dff/D ratio, the three
  softmaxes, how to read the weights, the hybrid BP head taxonomy
- **Part IV — Grounding and Hallucination**: grounded vs. ungrounded
  transformers, what hallucination means in BP terms
- **Part V — Convergent Evidence**: mechanistic interpretability findings,
  related work across five research traditions

## Building

Requires a standard LaTeX installation with `pdflatex` and `bibtex`.

    make

Produces `index.pdf`.