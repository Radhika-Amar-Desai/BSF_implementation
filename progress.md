# Implementation Progress and Major Changes

This document records the implementation progress, design decisions, and modifications made while reproducing the results from the Block Sparse Featurizer (BSF) paper.

## Components Adapted from the Original Implementation

The following components were adapted from the official Block Sparse Featurizer implementation:

- **`data.py`**
  - Based on the original implementation from the [Block Sparse Featurizer repository](https://github.com/goodfire-ai/block-sparse-featurizer).
  - Modified to extract DINOv2 activations from the provided `rabbit.npz` dataset instead of the original activation format.

## Original Implementation

The following components were implemented independently based on the methodology described in the research paper:

- **`vanilla_bsf.py`**
  - Implemented from scratch by following the algorithms and descriptions presented in the paper, without directly copying the reference implementation.