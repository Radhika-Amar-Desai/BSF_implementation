# Vanilla BSF

## Overall Pipeline

```text
Image
    ↓
Foundation Model (DINOv2 / DINOv3)
    ↓
Patch Activations
(N patches × d-dimensional embedding)
    ↓
Linear Encoder (W) And Bias Term (b)
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
```

The encoder maps every patch embedding into a high-dimensional latent space. The latent vector is partitioned into fixed-size blocks, and only the **Top-K blocks (based on their L2 norm)** are retained while the remaining blocks are zeroed out.

The decoder reconstructs the original patch embedding using only these active blocks. Since each decoder block is normalized to unit Frobenius norm, the block coefficients directly determine the contribution of each learned concept to the reconstruction.

After training, the primary outputs of interest are:

- **Latent block activations (`z_blocks`)** – indicate which concepts are active for each patch.
- **Decoder blocks (`D`)** – represent the learned concept dictionary used to reconstruct the embedding space.

Every reconstructed embedding can therefore be interpreted as a linear combination of the active concept manifolds.

---

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

- **3D Manifold:** *How does a learned concept vary across the dataset?*
- **Heatmap Overlay:** *Where does that concept appear in an image, and which semantic variation is present at each patch?*