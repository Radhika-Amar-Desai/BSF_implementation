# Implementation Progress and Major Changes

This document records the implementation progress, design decisions, and modifications made while reproducing the results from the Block Sparse Featurizer (BSF) paper.

## Components Adapted from the Original Implementation

The following components were adapted from the official Block Sparse Featurizer implementation:

- **`data.py`**
  - Based on the original implementation from the [Block Sparse Featurizer repository](https://github.com/goodfire-ai/block-sparse-featurizer).
  - Modified to extract DINOv2 activations from the provided `rabbit.npz` dataset instead of the original activation format.

## Implementation and Corrections

The following components were implemented independently based on the methodology described in the research paper:

- **`vanilla_bsf.py`**
  - Implemented from scratch by following the algorithms and descriptions presented in the paper, without directly copying the reference implementation.

- **`improved_vanilla_bsf.py`**
  - What I corrected:
    - Block Normalization of D. This is required because we judge top-k block selection based on norm of block co-ordinates mentioned in z. since the final equation is z * D, the model while training has no incentive choosing (z/10) * (10 * Dg) or z * Dg, but such scaling affects our top-k block selection.
    - Intialization trick. We need to initialize encoder as D. This is because when encoder is random, the norm of blocks may be tiny. In this architecture we select blocks according to norm and only the active blocks get updated during training, because of this, the blocks with small initial norm never get a chance to update themselves. Thus we initialize encoder as transpose of decoder and later encoder-decoder are untied. If decoder has meaningful concepts, encoder would discover it since it has same values as decoder and the operation $W^T.x$ is equivalent to dot product with each element of W. This increases the chance of selection of a block.

- **`\notebooks`**
  - What I corrected:
    - Proper pre-processing of foundation model embeddings and scaling of embeddings. Embeddings need to be scaled to $ \sqrt{d} $ where $d = dimensions\_of\_foundation\_model$. This is needed so that random intialization operations work smoothly. Random initializations try to keep output variance roughly equal to 1 to present exploding and vanishing gradients and assume the input has a variance of $ \sqrt{d} $.

- **`improved_grassmann_bsf.py`**
  - What I corrected:
    - Applying QR decomposition at every step instead of every 20 steps. In paper, applying QR decomposition at every 20 steps in mentioned but in the implementation QR decomposition is applied at every step and is differentiable.