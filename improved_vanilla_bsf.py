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
    reconstruction : (B, input_dim)
    z_blocks       : (B, G, b)
    z              : (B, latent_dim)
    active_mask    : (B, G)
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

        self.input_dim = W.in_features
        self.latent_dim = W.out_features
        self.num_blocks = self.latent_dim // block_size

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Normalize every decoder block to unit Frobenius norm.

        Decoder weight shape:
            (input_dim, latent_dim)

        Reshape into
            (num_blocks, block_size, input_dim)
        """

        weight = self.D.weight.data.T

        weight = weight.view(
            self.num_blocks,
            self.block_size,
            self.input_dim,
        )

        block_norms = (
            weight.flatten(start_dim=1)
            .norm(p=2, dim=1, keepdim=True)
            .clamp(min=1e-8)
        )

        weight = weight / block_norms.unsqueeze(-1)

        self.D.weight.data.copy_(
            weight.view(
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

    model = Vanilla_BSF(
        W=encoder,
        D=decoder,
        bias=bias,
        block_size=block_size,
        top_k=top_k,
    )

    #########################################
    # Initialization
    #########################################

    with torch.no_grad():

        # Random decoder initialization
        nn.init.normal_(
            model.D.weight,
            mean=0.0,
            std=0.02,
        )

        # Normalize decoder blocks
        model.normalize_decoder()

        # Encoder initialized as decoder transpose
        model.W.weight.copy_(
            model.D.weight.T
        )

    return model