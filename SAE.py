import torch
import torch.nn as nn


class Sparse_Autoencoder(nn.Module):
    """
    Standard Sparse Autoencoder (SAE)

    Input
    -----
    x : (B, input_dim)

    Output
    ------
    reconstruction : (B, input_dim)
    z              : (B, latent_dim)
    """

    def __init__(
        self,
        W,
        D,
        bias,
    ):
        super().__init__()

        self.W = W
        self.D = D
        self.bias = bias

    def forward(self, x):

        #########################################
        # Encode
        #########################################

        z = torch.relu(self.W(x) + self.bias)

        #########################################
        # Decode
        #########################################

        reconstruction = self.D(z)

        return {
            "reconstruction": reconstruction,
            "z": z,
        }


def build_model(
    input_dim=768,
    latent_dim=768,
):

    #########################################
    # Encoder
    #########################################

    encoder = nn.Linear(
        input_dim,
        latent_dim,
        bias=False,
    )

    #########################################
    # Decoder
    #########################################

    decoder = nn.Linear(
        latent_dim,
        input_dim,
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

    model = Sparse_Autoencoder(
        W=encoder,
        D=decoder,
        bias=bias,
    )

    return model