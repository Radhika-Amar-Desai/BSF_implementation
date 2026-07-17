# Vanilla BSF

## Assumptions And Design

Vanilla BSF is motivated by the hypothesis that **visual concepts are not isolated directions but low-dimensional manifolds embedded within the representation space of a foundation model (e.g., DINOv2/DINOv3).**

An activation is assumed to be a sparse sum of contributions from a few active concepts: $x = \sum_{i \in S} m_i$ where each contribution \(m_i\) lies on a concept manifold that is approximated by a low-dimensional linear subspace.

The model therefore learns:

- **Decoder \(D\)**: Learns a dictionary of subspaces. Each decoder block \(D_g\) defines the **basis vectors** of one low-dimensional subspace in the original DINO representation space.
- **Encoder \(W\)**: Learns the coordinates \(z_g\) of an activation within each learned subspace.
- **Block Top-K Projection**: Activates only the \(k\) most relevant concept subspaces.
- **Reconstruction**: Each concept contribution is reconstructed as $(m_g = z_g D_g)$, and the final representation is the sum of all active contributions.

## Overall Pipeline

````text
DINO Patch Embedding (x)
        │
        ▼
Linear Encoder (W, b)
        │
        ▼
Latent Code (z)
        │
Partition into blocks
        │
        ▼
Top-K Block Selection
        │
        ▼
Coordinates of Active Concepts
        │
        ▼
Decoder Blocks (D)
(learned bases of concept subspaces)
        │
        ▼
Per-concept Contributions
    m_g = z_g D_g
        │
        ▼
Sum of Active Contributions
        │
        ▼
Reconstructed DINO Embedding

## Overall Pipeline

```text
Image
    ↓
Foundation Model (DINOv2 / DINOv3)
    ↓
Patch Activations
(N patches × d-dimensional embeddings)
    ↓
Pre-processing
    • Subtract ImageNet per-position mean (POS_MEAN)
    • Flatten all patch embeddings into a single matrix
    • Compute dataset RMS norm
    • Rescale activations so RMS norm = √d
    ↓
Linear Encoder (W) and Bias Term (b)
    ↓
Latent Representation (z)
    ↓
Partition into blocks of size b
    ↓
Compute L2 norm of each block
    ↓
Keep Top-K blocks (Block Projection)
    ↓
Flatten sparse latent representation
    ↓
Linear Decoder (D)
(Block-normalized: each decoder block has unit Frobenius norm)
    ↓
Reconstructed Patch Activation
````

Before training the Block Sparse Featurizer, the patch embeddings undergo a preprocessing stage. First, the **ImageNet per-position mean (`POS_MEAN`)** is subtracted from every patch embedding to remove the average positional bias learned by the vision transformer. The patch embeddings from all images are then flattened into a single dataset of activation vectors.

Next, the dataset is **globally rescaled**. Let \(x_i \in \mathbb{R}^d\) denote the \(i^{\text{th}}\) patch embedding. The root mean squared (RMS) norm of the dataset is computed as

$$
\text{RMS}(x)
=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}\|x_i\|_2^2},
$$

and every activation is transformed as

$$
x_i'
=
x_i\,
\frac{\sqrt{d}}{\text{RMS}(x)}.
$$

This is **not** an L2 normalization of individual embeddings. Every activation is multiplied by the same global scaling factor, preserving the relative magnitudes between embeddings while ensuring that the dataset has an RMS norm of approximately \(\sqrt{d}\). Consequently, each feature dimension has approximately unit variance, matching the scale typically assumed by neural network initialization schemes and making optimization more stable.

The encoder then maps every preprocessed patch embedding into a high-dimensional latent space. The latent vector is partitioned into fixed-size blocks, and only the **Top-K blocks (based on their L2 norm)** are retained while the remaining blocks are zeroed out.

The decoder reconstructs the original patch embedding using only these active blocks. Since each decoder block is normalized to unit Frobenius norm, the block coefficients directly determine the contribution of each learned concept to the reconstruction.

After training, the primary outputs of interest are:

- **Latent block activations (`z_blocks`)** – indicate which concepts are active for each patch.
- **Decoder blocks (`D`)** – represent the learned concept dictionary used to reconstruct the embedding space.

## Every reconstructed embedding can therefore be interpreted as a linear combination of the active concept manifolds.

# Visualization Pipeline

The learned concepts are visualized using **two complementary views**:

1. **3D Concept Manifold**
2. **Heatmap Overlay on Images**

Together, these explain both **what a concept represents** and **where it appears**.

## 1. 3D Concept Manifold

For a selected concept (decoder block):

1. Collect every image patch where the concept is active.
2. Decode the latent block back into the original DINO embedding space using the corresponding decoder block:

   ```
   Contribution = z_block × Decoder Block
   ```

3. This produces one **768-dimensional contribution vector** for every activated patch.
4. Apply **PCA** to project these contribution vectors into 3 dimensions.
5. Visualize each patch as a point in the resulting 3D manifold.

### Interpretation

- **Position (x, y, z):** Represents the semantic variation of the concept. Similar patches appear close together.
- **Color (RGB):** Encodes the direction of the point in the PCA space and is shared with the image overlay to identify corresponding semantic variations.
- **Transparency:** Indicates the strength of the concept (computed from the projected manifold in the current implementation).

The manifold therefore illustrates **how a learned concept varies across the entire dataset**.

---

## 2. Heatmap Overlay

For the same concept:

1. Select the images in which the concept is most active.
2. Compute the contribution vector for every patch using the corresponding decoder block.
3. Project every contribution vector onto the same PCA basis computed for the manifold.
4. Overlay the patch grid on the original image.

### Interpretation

- **Color (RGB):** Represents the semantic variation of the concept using the same color mapping as the manifold.
- **Transparency:** Represents the magnitude of the **original 768-dimensional contribution vector**, indicating how strongly the concept contributes to that patch.

The heatmap therefore shows **where the concept appears within an image**, while the shared color mapping identifies **which region of the concept manifold each patch belongs to**.

---

## Summary

The two visualizations answer complementary questions:

- **3D Manifold:** _How does a learned concept vary across the dataset?_
- **Heatmap Overlay:** _Where does that concept appear in an image, and which semantic variation is present at each patch?_
