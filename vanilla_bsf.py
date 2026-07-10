import torch
import torch.nn as nn


class Vanilla_BSF(nn.Module):
    """
    Vanilla Block Sparse Featurizer

    Input
    -----
    x : (B, input_dim)

    Output
    ------
    reconstruction : (B,input_dim)
    z_blocks       : (B,G,b)
    z              : (B,latent_dim)
    active_mask    : (B,G)
    """

    def __init__(
        self,
        W,
        D,
        bias,
        block_size,
        top_k,
    ):
        super().__init__()

        self.W = W
        self.D = D
        self.bias = bias

        self.block_size = block_size
        self.top_k = top_k

    def block_projection(self, z):

        B, latent_dim = z.shape

        assert latent_dim % self.block_size == 0

        num_blocks = latent_dim // self.block_size

        #########################################
        # (B,latent) -> (B,G,b)
        #########################################

        z_blocks = z.view(B, num_blocks, self.block_size)

        #########################################
        # Block norms
        # (B,G)
        #########################################

        block_norms = torch.norm(
            z_blocks,
            p=2,
            dim=2,
        )

        #########################################
        # Top-k blocks
        #########################################

        _, topk_indices = torch.topk(
            block_norms,
            self.top_k,
            dim=1,
        )

        #########################################
        # Binary support mask
        #########################################

        active_mask = torch.zeros_like(block_norms)

        active_mask.scatter_(
            1,
            topk_indices,
            1,
        )

        #########################################
        # Expand to (B,G,1)
        #########################################

        active_mask = active_mask.unsqueeze(-1)

        #########################################
        # Zero inactive blocks
        #########################################

        z_blocks = z_blocks * active_mask

        return z_blocks, active_mask.squeeze(-1)

    def forward(self, x):

        #########################################
        # Encode
        #########################################

        z = self.W(x) + self.bias

        #########################################
        # Block Projection
        #########################################

        z_blocks, active_mask = self.block_projection(z)

        #########################################
        # Flatten
        #########################################

        z = z_blocks.flatten(start_dim=1)

        #########################################
        # Decode
        #########################################

        reconstruction = self.D(z)

        return {
            "reconstruction": reconstruction,
            "z": z,
            "z_blocks": z_blocks,
            "active_mask": active_mask,
        }


def build_model(
    input_dim=768,
    num_blocks=4096,
    block_size=4,
    top_k=16,
):

    latent_dim = num_blocks * block_size

    encoder = nn.Linear(
        input_dim,
        latent_dim,
        bias=False,
    )

    decoder = nn.Linear(
        latent_dim,
        input_dim,
        bias=False,
    )

    bias = nn.Parameter(
        torch.zeros(latent_dim)
    )

    model = Vanilla_BSF(
        W=encoder,
        D=decoder,
        bias=bias,
        block_size=block_size,
        top_k=top_k,
    )

    return model