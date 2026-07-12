import torch
import torch.nn as nn


class Grassmann_BSF(nn.Module):
    """
    Grassmannian Block Sparse Featurizer

    Changes from Vanilla BSF
    ------------------------
    - Tied encoder:
            z = gamma * x D^T
    - Decoder blocks are constrained to the Stiefel manifold
      using QR decomposition every forward pass.
    """

    def __init__(
        self,
        input_dim,
        num_blocks,
        block_size,
        top_k,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.top_k = top_k

        self.latent_dim = num_blocks * block_size

        # Raw decoder parameter
        # Shape: (num_blocks, input_dim, block_size)
        self.D_raw = nn.Parameter(
            torch.randn(
                num_blocks,
                input_dim,
                block_size,
            )
        )

        # gamma = exp(log_gamma)
        self.log_gamma = nn.Parameter(torch.zeros(()))

    def decoder_atoms(self):
        """
        Returns decoder matrix with orthonormal blocks.

        Output shape:
            (latent_dim, input_dim)
        """

        # QR decomposition for every block
        B, _ = torch.linalg.qr(
            self.D_raw,
            mode="reduced",
        )

        atoms = (
            B.permute(0, 2, 1)
            .reshape(
                self.latent_dim,
                self.input_dim,
            )
        )

        return atoms

    def block_projection(self, z):

        B, _ = z.shape

        z_blocks = z.view(
            B,
            self.num_blocks,
            self.block_size,
        )

        block_norms = torch.norm(
            z_blocks,
            dim=2,
        )

        _, topk_indices = torch.topk(
            block_norms,
            self.top_k,
            dim=1,
        )

        active_mask = torch.zeros_like(block_norms)

        active_mask.scatter_(
            1,
            topk_indices,
            1.0,
        )

        z_blocks = z_blocks * active_mask.unsqueeze(-1)

        return z_blocks, active_mask

    def forward(self, x):

        ####################################
        # Decoder atoms (QR every forward)
        ####################################

        atoms = self.decoder_atoms()

        ####################################
        # Encode
        ####################################

        gamma = torch.exp(self.log_gamma)

        z = gamma * (x @ atoms.T)

        ####################################
        # Block projection
        ####################################

        z_blocks, active_mask = self.block_projection(z)

        ####################################
        # Flatten
        ####################################

        z = z_blocks.flatten(start_dim=1)

        ####################################
        # Decode
        ####################################

        reconstruction = z @ atoms

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

    model = Grassmann_BSF(
        input_dim=input_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        top_k=top_k,
    )

    with torch.no_grad():

        # Initial orthogonalization
        B, _ = torch.linalg.qr(
            model.D_raw,
            mode="reduced",
        )

        model.D_raw.copy_(B)

        model.log_gamma.zero_()

    return model