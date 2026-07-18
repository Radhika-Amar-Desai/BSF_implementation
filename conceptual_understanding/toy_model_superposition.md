# Toy Model Superposition

* Primarily to test whether block sparsity is the valid prior or not.

* The authors construct toy concept manifolds in the shapes like sphere, donut, a disk etc. and all points are constructed from a point from each of these manifolds.

* Next, BSF architectures, SAE, SMixAE and MFA are trained on the synthetically generated data points.

* We find what blocks fire the most when each primitive is present. We can pass the points from sphere for example to see what block fires the most.

* Now, each block is assigned to each primitve and using the co-ordinates $Z_g$ and decoder $D_g$ we try and reconstruct the contribution provided by the particular block and see how well it reconstructs its corresponding primitive using $R^2$ metrics.

## Visualization Guide

* For each contribution $m_i$ we perform PCA and that determines its positional co-ordinates, for ground truth we perform PCA and assign the corresponding values to R, G and B channels to see how well the featurizer has reconstructed the primitive.

## SMixAE

x -> Linear -> LeakyReLU or ReLU -> Linear -> Top-k experts -> Linear -> Linear -> $x_{reconstructed}$

Here, replacing ReLU helps becoz ReLU gives the contributions a cone shaped since only postive values are considered.

## MFA

This model assumes a mixture of gaussian model, which says that any data point is produced by union of semantic concept subspaces rather than by their sum.

Contributions of each semantic concept subspace are assigned a probability score.

This is equivalent to gaussian mixture model.

The authors fix this by suming up contributions from various subspaces. So that each data point isn't recovered just from overlap or union of subspaces but by minwoski sum of minifolds.