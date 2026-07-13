import math

import torch
import torch.nn as nn


class MDL(nn.Module):
    """
    Minimum Description Length (Eq. 5)

    Returns

        total_bits
        support_bits
        code_bits
        residual_bits
        dictionary_bits
    """

    def __init__(
        self,
        num_blocks,
        block_size,
        dataset_tokens,
        residual_sigma=1.0,
        code_sigma=1.0,
    ):
        super().__init__()

        self.G = num_blocks
        self.b = block_size
        self.N = dataset_tokens

        self.residual_sigma = residual_sigma
        self.code_sigma = code_sigma

    ###########################################################
    # log2(n choose k)
    ###########################################################

    @staticmethod
    def log2_binomial(n, k):

        lg = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
        )

        return lg / math.log(2)

    ###########################################################
    # Support cost
    ###########################################################

    def support_bits(self, active_mask):

        # active blocks for each sample

        k = active_mask.sum(dim=1)

        bits = []

        for kk in k.tolist():
            bits.append(
                self.log2_binomial(
                    self.G,
                    int(kk),
                )
            )

        return torch.tensor(
            bits,
            device=active_mask.device,
        )

    ###########################################################
    # Code cost
    ###########################################################

    def code_bits(self, z):

        # Assume Gaussian entropy

        bits = (
            z.pow(2).sum(dim=1)
            /
            (2 * self.code_sigma ** 2 * math.log(2))
        )

        return bits

    ###########################################################
    # Residual cost
    ###########################################################

    def residual_bits(
        self,
        x,
        reconstruction,
    ):

        residual = x - reconstruction

        bits = (
            residual.pow(2).sum(dim=1)
            /
            (2 * self.residual_sigma ** 2 * math.log(2))
        )

        return bits

    ###########################################################
    # Dictionary cost
    ###########################################################

    def dictionary_bits(self, decoder):

        weight = decoder.weight

        bits = (
            weight.numel() * 32
        ) / self.N

        return torch.tensor(
            bits,
            device=weight.device,
        )

    ###########################################################

    def forward(
        self,
        x,
        output,
        decoder,
    ):

        support = self.support_bits(
            output["active_mask"]
        )

        code = self.code_bits(
            output["z"]
        )

        residual = self.residual_bits(
            x,
            output["reconstruction"],
        )

        dictionary = self.dictionary_bits(
            decoder
        )

        total = (
            support
            + code
            + residual
            + dictionary
        )

        return {
            "total": total.mean(),
            "support": support.mean(),
            "code": code.mean(),
            "residual": residual.mean(),
            "dictionary": dictionary,
        }