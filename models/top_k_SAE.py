import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """
    Top-k Sparse Autoencoder

    Encoder:
        z = ReLU(Wx + b)

    Sparsification:
        Keep only the k blocks with the largest L2 norms.

    Decoder:
        x_hat = Dz

    Parameters
    ----------
    input_dim : int
    num_blocks : int
    block_size : int
    top_k : int

    Shapes
    ------
    x               : (B, input_dim)
    z               : (B, latent_dim)
    z_blocks        : (B, num_blocks, block_size)
    active_mask     : (B, num_blocks)
    reconstruction  : (B, input_dim)
    """

    def __init__(
        self,
        input_dim=768,
        num_blocks=4096,
        block_size=1,
        top_k=64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.top_k = top_k

        self.latent_dim = num_blocks * block_size

        self.encoder = nn.Linear(
            input_dim,
            self.latent_dim,
            bias=False,
        )

        self.decoder = nn.Linear(
            self.latent_dim,
            input_dim,
            bias=False,
        )

        self.bias = nn.Parameter(
            torch.zeros(self.latent_dim)
        )

        self.reset_parameters()

    ###########################################################

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    ###########################################################

    def atoms(self):
        """
        Returns decoder atoms.

        Shape:
            (num_blocks, block_size, input_dim)
        """

        return self.decoder.weight.T.view(
            self.num_blocks,
            self.block_size,
            self.input_dim,
        )

    ###########################################################

    def encode(self, x):

        z = torch.relu(
            self.encoder(x) + self.bias
        )

        z_blocks, active_mask = self.block_projection(z)

        z = z_blocks.reshape(
            x.shape[0],
            self.latent_dim,
        )

        return z, z_blocks, active_mask

    ###########################################################

    def decode(self, z):

        return self.decoder(z)

    ###########################################################

    def block_projection(self, z):

        B = z.shape[0]

        z_blocks = z.view(
            B,
            self.num_blocks,
            self.block_size,
        )

        #######################################################
        # Block norms
        #######################################################

        block_norms = torch.norm(
            z_blocks,
            dim=-1,
        )

        #######################################################
        # Top-k blocks
        #######################################################

        _, top_idx = torch.topk(
            block_norms,
            k=self.top_k,
            dim=1,
        )

        #######################################################
        # Binary support mask
        #######################################################

        active_mask = torch.zeros_like(
            block_norms
        )

        active_mask.scatter_(
            1,
            top_idx,
            1.0,
        )

        #######################################################
        # Zero inactive blocks
        #######################################################

        z_blocks = (
            z_blocks
            * active_mask.unsqueeze(-1)
        )

        return z_blocks, active_mask

    ###########################################################

    def forward(self, x):

        z, z_blocks, active_mask = self.encode(x)

        reconstruction = self.decode(z)

        return {
            "reconstruction": reconstruction,
            "z": z,
            "z_blocks": z_blocks,
            "active_mask": active_mask,
        }


##############################################################


def build_model(
    input_dim=768,
    num_blocks=4096,
    block_size=1,
    top_k=64,
):

    return TopKSAE(
        input_dim=input_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        top_k=top_k,
    )