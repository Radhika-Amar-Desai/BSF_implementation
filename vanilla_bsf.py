import torch
import torch.nn as nn

class Vanilla_BSF(nn.Module):
    """
    Vanilla Block Sparse Featurizer (BSF)

    Input:
        x : (B, input_dim)

    Output:
        reconstruction : (B, input_dim)
        z_blocks       : (B, G, b)
        active_mask    : (B, G)
    """

    def __init__(self, W, D, bias, b):
        """
        Parameters
        ----------
        W : Encoder (typically nn.Linear)
        D : Decoder (typically nn.Linear)
        bias : Encoder bias
        b : Block size
        """
        self.W = W
        self.D = D
        self.bias = bias
        self.b = b

    def block_projection(self, z, k):
        """
        Implements Π_k from the paper.

        Parameters
        ----------
        z : (B, latent_dim)
            Encoder output

        k : int
            Number of active blocks

        Returns
        -------
        z_blocks : (B, G, b)
            Block-sparse latent code

        active_mask : (B, G)
            Binary mask indicating active blocks
        """

        B, latent_dim = z.shape

        assert latent_dim % self.b == 0, \
            "Latent dimension must be divisible by block size."

        G = latent_dim // self.b

        # --------------------------------------------------
        # Split latent vector into blocks
        # Shape : (B, G, b)
        # --------------------------------------------------
        z_blocks = z.view(B, G, self.b)

        # --------------------------------------------------
        # Compute L2 norm of every block
        # Shape : (B, G)
        # --------------------------------------------------
        block_norms = torch.norm(z_blocks, p=2, dim=2)

        # --------------------------------------------------
        # Top-k block selection
        # Shape : (B, k)
        # --------------------------------------------------
        _, topk_indices = torch.topk(block_norms, k, dim=1)

        # --------------------------------------------------
        # Binary mask of active blocks
        # Shape : (B, G)
        # --------------------------------------------------
        active_mask = torch.zeros_like(block_norms)

        active_mask.scatter_(1, topk_indices, 1)

        # --------------------------------------------------
        # Expand mask to block dimension
        # Shape : (B, G, 1)
        # --------------------------------------------------
        expanded_mask = active_mask.unsqueeze(-1)

        # --------------------------------------------------
        # Zero-out inactive blocks
        # Shape : (B, G, b)
        # --------------------------------------------------
        z_blocks = z_blocks * expanded_mask

        return z_blocks, active_mask

    def forward(self, x, k):
        """
        Forward pass

        Pipeline
        --------
        Input embedding
                ↓
        Linear Encoder
                ↓
        Block Projection (Π_k)
                ↓
        Linear Decoder
                ↓
        Reconstructed embedding
        """

        # --------------------------------------------------
        # Encode
        # Shape : (B, latent_dim)
        # --------------------------------------------------
        z = self.W(x) + self.bias

        # --------------------------------------------------
        # Block sparse projection
        # --------------------------------------------------
        z_blocks, active_mask = self.block_projection(z, k)

        # --------------------------------------------------
        # Flatten before decoder
        # Shape : (B, latent_dim)
        # --------------------------------------------------
        B = z.shape[0]
        z_flat = z_blocks.view(B, -1)

        # --------------------------------------------------
        # Decode
        # Shape : (B, input_dim)
        # --------------------------------------------------
        reconstruction = self.D(z_flat)

        return {
            "reconstruction": reconstruction,
            "z_blocks": z_blocks,
            "z": z_flat,
            "active_mask": active_mask,
        }


"""
Overall Pipeline

Image
   ↓
Foundation Model / Backbone (DINOv3, CNN, ViT, etc.)
   ↓
Feature Embedding x
   ↓
Linear Encoder
   ↓
Latent Code z
   ↓
Split into G blocks of size b
   ↓
Compute ||z_g||₂ for every block
   ↓
Keep Top-k blocks (Π_k)
   ↓
Block Sparse Code z_blocks
   ↓
Flatten
   ↓
Linear Decoder
   ↓
Reconstructed Embedding x̂


Returned Outputs

reconstruction : reconstructed embedding x̂
z_blocks       : latent code grouped into blocks (contains the internal coordinates)
z              : flattened latent code
active_mask    : binary indicator of which concept blocks are active
"""