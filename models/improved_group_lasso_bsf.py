
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockJumpReLU(torch.autograd.Function):
    """
    Hard block gate with a straight-through estimator (STE).

    Forward:
        gate = 1 if ||z_g|| > theta_g else 0

    Backward:
        Uses a rectangle-kernel pseudo-derivative for theta.
        No gradient is propagated through the block norm.
    """

    @staticmethod
    def forward(ctx, block_norms, theta, bandwidth):
        ctx.save_for_backward(block_norms, theta)
        ctx.bandwidth = float(bandwidth)
        return (block_norms > theta).to(block_norms.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        block_norms, theta = ctx.saved_tensors

        kernel = (
            (block_norms - theta).abs()
            <= ctx.bandwidth / 2
        ).to(block_norms.dtype) / ctx.bandwidth

        grad_theta = -(grad_output * kernel).sum(dim=0)

        return None, grad_theta, None


class Group_Lasso_BSF(nn.Module):
    """
    Group Lasso BSF (Repository / Block JumpReLU version)

    Differences from the paper version
    ---------------------------------
    * Hard block gating instead of soft-thresholding.
    * Magnitudes of active blocks are preserved.
    * One learnable threshold per block.
    * Thresholds are learned with a Straight-Through Estimator.
    * Intended to be trained with an L0 penalty on active blocks.
    """

    def __init__(
        self,
        W,
        D,
        bias,
        block_size,
        target_l0=16,
        gain=10.0,
    ):
        super().__init__()

        self.W = W
        self.D = D
        self.bias = bias

        self.block_size = block_size
        self.target_l0 = target_l0
        self.gain = gain

        self.input_dim = W.in_features
        self.latent_dim = W.out_features
        self.num_blocks = self.latent_dim // block_size

        self.raw_theta = nn.Parameter(
            torch.zeros(self.num_blocks)
        )

        self.register_buffer(
            "bandwidth",
            torch.ones(())
        )

        self.register_buffer(
            "inited",
            torch.zeros((), dtype=torch.bool)
        )

    @torch.no_grad()
    def normalize_decoder(self):

        weight = self.D.weight.data.T

        weight = weight.view(
            self.num_blocks,
            self.block_size,
            self.input_dim,
        )

        norms = (
            weight.flatten(start_dim=1)
            .norm(dim=1, keepdim=True)
            .clamp(min=1e-8)
        )

        weight = weight / norms.unsqueeze(-1)

        self.D.weight.data.copy_(
            weight.view(
                self.latent_dim,
                self.input_dim,
            ).T
        )

    def theta(self):
        return F.softplus(
            self.gain * self.raw_theta
        )

    @torch.no_grad()
    def init_theta(self, block_norms):

        q = 1.0 - self.target_l0 / self.num_blocks

        threshold = torch.quantile(
            block_norms.flatten(),
            q,
        ).clamp_min(1e-3)

        self.raw_theta.copy_(
            torch.log(torch.expm1(threshold))
            / self.gain
        )

        self.bandwidth.copy_(
            block_norms.std() * 0.5 + 1e-6
        )

        self.inited.fill_(True)

    def block_gate(self, block_norms):

        if self.training and not bool(self.inited):
            self.init_theta(block_norms)

        return BlockJumpReLU.apply(
            block_norms,
            self.theta(),
            float(self.bandwidth),
        )

    def block_jump_relu(self, z):

        B = z.shape[0]

        z_blocks = z.view(
            B,
            self.num_blocks,
            self.block_size,
        )

        block_norms = z_blocks.norm(dim=2)

        gate = self.block_gate(block_norms)

        z_blocks = z_blocks * gate.unsqueeze(-1)

        return z_blocks, gate

    def forward(self, x):

        z = self.W(x) + self.bias

        z_blocks, gate = self.block_jump_relu(z)

        z = z_blocks.flatten(start_dim=1)

        reconstruction = self.D(z)

        return {
            "reconstruction": reconstruction,
            "z": z,
            "z_blocks": z_blocks,
            "gate": gate,
        }


def build_model(
    input_dim=768,
    num_blocks=4096,
    block_size=4,
    target_l0=16,
):

    latent_dim = num_blocks * block_size

    decoder = nn.Linear(
        latent_dim,
        input_dim,
        bias=False,
    )

    encoder = nn.Linear(
        input_dim,
        latent_dim,
        bias=False,
    )

    bias = nn.Parameter(
        torch.zeros(latent_dim)
    )

    model = Group_Lasso_BSF(
        W=encoder,
        D=decoder,
        bias=bias,
        block_size=block_size,
        target_l0=target_l0,
    )

    with torch.no_grad():

        model.D.weight.copy_(
            torch.randn_like(model.D.weight)
        )

        model.normalize_decoder()

        model.W.weight.copy_(
            model.D.weight.T
        )

    return model
