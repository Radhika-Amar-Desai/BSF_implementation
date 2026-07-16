import torch
import torch.nn as nn


class Group_Lasso_BSF(nn.Module):
    """
    Group Lasso Block Sparse Featurizer

    Differences from Vanilla BSF
    ----------------------------
    - Same linear encoder and decoder.
    - Same latent bias.
    - Hard Top-K projection is replaced by block soft-thresholding.
    - One learnable threshold per block.
    - Sparsity is enforced through the Group Lasso loss during training.
    """

    def __init__(
        self,
        W,
        D,
        bias,
        block_size,
    ):
        super().__init__()

        self.W = W
        self.D = D
        self.bias = bias

        self.block_size = block_size

        self.input_dim = W.in_features
        self.latent_dim = W.out_features
        self.num_blocks = self.latent_dim // block_size

        #########################################
        # One learnable threshold per block
        # theta_g = exp(log_theta_g) > 0
        #########################################

        self.log_theta = nn.Parameter(
            torch.zeros(self.num_blocks)
        )

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Normalize every decoder block to unit Frobenius norm.
        """

        weight = self.D.weight.data.T

        weight = weight.view(
            self.num_blocks,
            self.block_size,
            self.input_dim,
        )

        block_norms = (
            weight.flatten(start_dim=1)
            .norm(dim=1, keepdim=True)
            .clamp(min=1e-8)
        )

        weight = weight / block_norms.unsqueeze(-1)

        self.D.weight.data.copy_(
            weight.view(
                self.latent_dim,
                self.input_dim,
            ).T
        )

    def block_soft_threshold(self, z):

        B = z.shape[0]

        z_blocks = z.view(
            B,
            self.num_blocks,
            self.block_size,
        )

        #########################################
        # Block norms
        #########################################

        norms = torch.norm(
            z_blocks,
            dim=2,
            keepdim=True,
        ).clamp(min=1e-8)

        #########################################
        # One threshold per block
        #########################################

        theta = torch.exp(
            self.log_theta
        ).view(
            1,
            self.num_blocks,
            1,
        )

        #########################################
        # Block soft-threshold
        #########################################

        shrink = torch.clamp(
            1.0 - theta / norms,
            min=0.0,
        )

        z_blocks = z_blocks * shrink

        return z_blocks

    def forward(self, x):

        #########################################
        # Encode
        #########################################

        z = self.W(x) + self.bias

        #########################################
        # Block soft-threshold
        #########################################

        z_blocks = self.block_soft_threshold(z)

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
        }


def build_model(
    input_dim=768,
    num_blocks=4096,
    block_size=4,
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
    # Encoder
    #########################################

    encoder = nn.Linear(
        input_dim,
        latent_dim,
        bias=False,
    )

    #########################################
    # Latent bias
    #########################################

    bias = nn.Parameter(
        torch.zeros(latent_dim)
    )

    #########################################
    # Build model
    #########################################

    model = Group_Lasso_BSF(
        W=encoder,
        D=decoder,
        bias=bias,
        block_size=block_size,
    )

    #########################################
    # Initialization
    #########################################

    with torch.no_grad():

        # Random decoder initialization
        model.D.weight.copy_(
            torch.randn_like(
                model.D.weight
            )
        )

        # Normalize decoder blocks
        model.normalize_decoder()

        # Initialize encoder as decoder transpose
        model.W.weight.copy_(
            model.D.weight.T
        )

        # theta_g = exp(0) = 1
        model.log_theta.zero_()

    return model