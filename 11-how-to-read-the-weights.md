# How to Read the Weights: A Field Guide

## The Question

If the transformer is a Bayesian network, you should be able to look at its weights and see the Bayes net. This file is a practical guide to what you would actually look for.

## What to Look For in Attention Weights

### The Q and K Matrices

In the formal BP construction, Q and K matrices have projectDim structure:

    W_Q = W_K = projectDim(d)

This is a rank-1 matrix with a single nonzero entry: 1 at position (d, d). As a Q matrix, it projects the query to dimension d. As a K matrix, it projects the key to dimension d. The dot product peaks when both tokens have the same value in dimension d — index matching.

What to look for empirically: low-rank Q and K matrices, with weight concentrated on a small number of dimensions. The psychic hypothesis: sharp routing heads should show approximately rank-1 Q·K structure, with the dominant singular vector corresponding to the "query" dimension.

Sharp attention = low-rank Q/K = index matching = clean gather step.
Diffuse attention = higher-rank Q/K = multiple dimensions contributing = approximate aggregation.

### The V Matrix

In the formal BP construction:

    W_V = crossProject(s, d)

This is a rank-1 matrix with a single nonzero entry: 1 at position (d, s). It reads from source dimension s and writes to destination dimension d. As a value projection, it copies one dimension of the attended token to one dimension of the output.

What to look for: rank-1 or low-rank V matrices, with the nonzero entry connecting a specific source dimension (where the neighbor belief is stored) to a specific destination dimension (the scratch slot in the residual stream).

### The OV Circuit

Elhage et al. (2021) showed that the product W_O · W_V (the "OV circuit") is what determines what the head copies. For a BP head, W_O · W_V should be a rank-1 matrix that copies dimension s to dimension d in the full D-dimensional residual stream.

This is a falsifiable prediction: sharp routing heads should have approximately rank-1 OV circuits. Diffuse heads should have higher-rank OV circuits.

## What to Look For in FFN Weights

### The W1 Matrix (Keys)

Each row of W1 is a "key" — a direction in the residual stream that triggers this hidden unit. In the formal BP construction, each key should have large weights only in the dimensions where attention wrote neighbor beliefs (dims 4 and 5 in the 8-dimensional construction).

For a production model with D=4096 and superposition: the keys should be sparse in their activation pattern (each unit fires for a small fraction of inputs) even if they are dense in weight. The effective sparsity is in the activations, not the weights.

What to look for: the activation frequency of FFN hidden units. BP predicts sparse activations — each unit is a specific concept detector that fires rarely. The Anthropic SAE work confirms this: FFN activations are highly sparse.

### The W2 Matrix (Values)

Each column of W2 is a "value" — what gets written to the residual stream when a hidden unit fires. In the formal BP construction, each value should update the specific dimension holding the belief being computed.

For a production model: W2 columns should correspond to specific directions in the residual stream that SAEs can identify as meaningful features. ROME confirms this: editing specific W2 columns at specific layers changes specific factual associations.

### The Factual Layer Structure

ROME found that factual associations are stored in specific middle layers (layers 13-17 for GPT-2 XL, approximately 1/3 to 1/2 of the way through the network). Earlier layers handle lower-level patterns; later layers handle abstract reasoning.

In BP terms: the first few layers do feature extraction (early rounds, gathering local evidence). The middle layers store factual associations (the factor potentials for common knowledge). The later layers do higher-order reasoning (applying the factual knowledge to the specific query).

## What the Psychic Experiments Found

The psychic repo classified GPT-2 small heads by type:

    boolean-AND (semantic): 9 heads, layers 0-4
    hub (positional-AND):   103 heads, layers 1-11
    continuous-OR:          28 heads, layers 0-2 and layer 11
    mixed:                  8 heads

Classification method: per-prompt entropy of attention distribution, max attention weight, previous-token attention, first-token attention, diagonal attention. No ground truth labels — fully unsupervised.

The boolean-AND heads concentrate in early layers because that is where structured cross-token reasoning first appears. The hub heads dominate the later layers because once local features are extracted, the model assembles the global context through position 0.

### The Predicted Structure for BP Verification

If a head is a true semantic-AND head (BP gather step), it should show:
1. Sharp attention (low entropy, high max weight)
2. Approximately rank-1 QK matrix (projectDim structure)
3. Approximately rank-1 OV circuit (crossProject structure)
4. Consistent attention target across prompts with the same syntactic structure

Properties 1 is checked by psychic. Properties 2-4 are not yet checked — they are predictions that could be verified by extending the psychic analysis to weight inspection.

## The Experiment That Would Confirm It

The definitive confirmation would be:

1. Take a sharp-attention head identified by psychic
2. Compute W_Q, W_K, W_V matrices
3. Compute the QK circuit (W_Q · W_K^T) and the OV circuit (W_O · W_V)
4. Check if QK is approximately rank-1 (projectDim structure)
5. Check if OV is approximately rank-1 (crossProject structure)
6. Identify which dimensions QK projects to and which OV copies between
7. Check if those dimensions correspond to SAE features with meaningful interpretations

If the answer to 4-6 is yes, and if the dimensions in step 6 correspond to interpretable features in step 7, then you have directly observed the BP routing structure in a trained transformer.

The psychic repo is set up to do this — it has the weight loading and analysis infrastructure. The head classification is done. The next step is the weight decomposition.

## What "Seeing the Bayes Net" Would Actually Look Like

If the full verification succeeded, you would be able to say something like:

"Head 2 at layer 3 of GPT-2 small is a semantic-AND head. Its QK circuit has rank 1, projecting to dimension 47 of the residual stream. Its OV circuit has rank 1, copying from dimension 47 to dimension 83. Dimension 47 corresponds to SAE feature 1024 ('subject of the current clause'). Dimension 83 corresponds to SAE feature 2891 ('entity being referenced'). This head implements the gather step for the 'subject reference' factor in the implicit factor graph — it copies the current clause's subject into the slot used by the next layer's FFN to compute coreferent resolution."

That is what seeing the Bayes net in the transformer would look like. The psychic and interp repos have laid the groundwork. The connection between the weight structure, the attention behavior, and the SAE features is the experiment that would close the loop.