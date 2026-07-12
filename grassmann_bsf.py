
import torch
import torch.nn as nn


class Grassmann_BSF(nn.Module):
    """
    Grassmannian Block Sparse Featurizer

    Differences from Vanilla BSF
    ----------------------------
    - No independent encoder
    - No latent bias
    - Encoder is tied to decoder:
            z = gamma * x D^T
    - Decoder blocks are constrained to be orthonormal
      using QR projection onto the Stiefel manifold.
    """

    def __init__(
        self,
        D,
        block_size,
        top_k,
    ):
        super().__init__()

        self.D = D

        self.log_gamma = nn.Parameter(torch.zeros(()))

        self.block_size = block_size
        self.top_k = top_k

        self.input_dim = D.out_features
        self.latent_dim = D.in_features
        self.num_blocks = self.latent_dim // block_size

    @torch.no_grad()
    def project_decoder_to_stiefel(self):
        """
        Project every decoder block onto the Stiefel manifold.

        Each block becomes an orthonormal basis.
        """

        weight = self.D.weight.data.T

        weight = weight.view(
            self.num_blocks,
            self.block_size,
            self.input_dim,
        )

        for g in range(self.num_blocks):

            block = weight[g]

            # QR on transpose because basis vectors are rows
            Q, _ = torch.linalg.qr(
                block.T,
                mode="reduced",
            )

            weight[g] = Q.T

        self.D.weight.data.copy_(
            weight.reshape(
                self.latent_dim,
                self.input_dim,
            ).T
        )

    def block_projection(self, z):

        B, latent_dim = z.shape

        z_blocks = z.view(
            B,
            self.num_blocks,
            self.block_size,
        )

        block_norms = torch.norm(
            z_blocks,
            p=2,
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

        #########################################
        # Encode (tied encoder)
        #########################################

        gamma = torch.exp(self.log_gamma)

        # D.weight : (input_dim, latent_dim)
        z = gamma * (x @ self.D.weight)

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

    #########################################
    # Decoder
    #########################################

    decoder = nn.Linear(
        latent_dim,
        input_dim,
        bias=False,
    )

    #########################################
    # Build model
    #########################################

    model = Grassmann_BSF(
        D=decoder,
        block_size=block_size,
        top_k=top_k,
    )

    #########################################
    # Initialization
    #########################################

    with torch.no_grad():

        model.D.weight.copy_(
            torch.randn_like(model.D.weight)
        )

        model.project_decoder_to_stiefel()

        model.log_gamma.fill_(0.0)

    return model

