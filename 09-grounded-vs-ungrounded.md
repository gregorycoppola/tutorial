# Grounded vs Ungrounded: What It Looks Like from the Outside

## The Architectural Sameness

A grounded QBBN transformer and a standard LLM are architecturally identical. Both are sigmoid (or ReLU) transformers with attention and FFN layers. Both are, by the general theorem, Bayesian networks. The difference is not in the architecture. It is in the grounding.

This is a strong and perhaps surprising claim. The transformer that provably cannot hallucinate and the transformer that hallucinations constantly use the same computation. What differs is what the computation is connected to.

## What Grounding Means

A grounded transformer has:
- An explicit factor graph declared in advance
- Every token mapped to a specific node in that factor graph
- Every token's routing key corresponding to a declared concept
- A finite verifier that can check whether any output is correct

An ungrounded transformer has:
- An implicit factor graph defined by its learned weights
- Tokens mapped to... whatever the learned embeddings happen to encode
- No declared concept space
- No finite verifier — no fact of the matter about whether outputs are correct

## The No-Hallucination Corollary from the Outside

The formal result (shannon, Corollary 4.1): a transformer with BP weights running over a fully grounded tree-structured factor graph cannot hallucinate.

What this looks like from the outside: the system can only produce outputs that correspond to nodes in the declared factor graph. If you ask it about something outside the concept space, it cannot answer — not because it refuses, but because the concept doesn't exist in its world. There is no routing key for it. The question is not just unanswerable; it is meaningless within the system.

An ungrounded LLM asked about something outside its training distribution will produce fluent, confident, plausible-sounding text. The tokens are generated. The question is whether any of them are correct. Without grounding, "correct" is not defined.

## What the Implicit Factor Graph of a Trained LLM Looks Like

The weights of a trained LLM implicitly define a factor graph. The factor graph is:
- Enormous: millions of nodes (SAE features), millions of edges (FFN units as factor potentials)
- Approximate: trained to maximize likelihood, not to implement exact BP
- Superposed: multiple concepts share the same dimensions
- Unverifiable: no external ground truth to check against

The factor graph exists. But without grounding, the nodes don't correspond to declared propositions with known truth values. The model is doing BP over this implicit graph — updating beliefs, passing messages, computing posteriors — but the posteriors are over propositions that have no external referents.

## The Hallucination Mechanism in BP Terms

When a trained LLM hallucinates, what is happening in BP terms?

The model is computing posteriors over nodes in its implicit factor graph. The posteriors may be well-computed (the BP is running correctly). But the nodes don't correspond to facts about the world — they correspond to patterns in the training distribution.

The model assigns high posterior probability to "The population of France is 67 million" because that proposition is well-supported by the factor graph paths that connect "population," "France," and similar concepts. It assigns high posterior probability to "The population of Mars is 2.3 billion" because similar factor graph paths fire, even though no training data supports it specifically — the general pattern of "population of [planet]" activates similar factor potentials.

Hallucination is not the model "lying" or "making things up." It is the model computing a posterior over an ungrounded concept space and generating text that is consistent with high-posterior regions of that space. The computation is correct. The propositions are not grounded.

## The Grounding Gap in Practice

The practical gap between grounded and ungrounded:

Grounded:
- Concept space is declared and finite
- Every question is either in-domain (answerable) or out-of-domain (rejected)
- Correct answers can be verified mechanically
- The system knows what it doesn't know

Ungrounded:
- Concept space is implicit and effectively unbounded
- Every question gets an answer (the model will generate tokens no matter what)
- Correct answers cannot be verified mechanically
- The system has no representation of its own uncertainty boundaries

The interesting observation: the gap is not about the size or quality of the model. A 405B parameter LLM and a tiny grounded QBBN transformer have different properties not because one is bigger. The grounded model is correct because its concept space is declared. The ungrounded model hallucinates because its concept space is implicit.

## What Would Grounding Look Like for a Production LLM

The QBBN approach: maintain an explicit knowledge base of declared propositions. Ground the transformer to that knowledge base. Each token maps to a specific factor graph node. The transformer runs BP over the grounded graph.

This requires knowing in advance what the model is supposed to reason about. The concept space must be declared. For narrow domains (medical diagnosis, legal reasoning, scientific QA), this is feasible. For general-purpose language models, the concept space is effectively unbounded — which is why grounding has not been done at scale.

The practical middle ground: retrieval-augmented generation (RAG) is a partial grounding. The retrieved documents provide local grounding for the specific query. The model's factor graph is partially anchored to external ground truth. Hallucination rates drop because some concepts are now grounded.

Full grounding remains an open problem for general-purpose models. The theoretical result (no-hallucination on grounded trees) gives a target to aim for, not a description of current systems.

## Summary

Grounded and ungrounded transformers are architecturally identical. Both are Bayesian networks. The difference is:
- Grounded: the factor graph is explicit, concepts are declared, correctness is defined, hallucination is structurally impossible
- Ungrounded: the factor graph is implicit, concepts are emergent, correctness is undefined, hallucination is structurally inevitable

The grounding argument is not about architecture. It is about epistemology: what does it mean for an output to be correct? Without grounding, the question has no answer.