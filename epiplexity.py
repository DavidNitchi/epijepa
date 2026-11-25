"""Epiplexity: how much of an embedding a ridge readout from a frozen random CNN can predict.

score(z, a) = 0.5 * log2 det(I + eta * W^T W),   W = a @ (z - mean(z))

where a is the ridge map from the reservoir's features on the same batch. A collapsed
embedding scores ~0; spreading predictable structure across more directions raises it.
Maximising this score, traded off against a view-invariance loss, is the whole method.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

RIDGE = 3.0  # readout regularisation
ETA = 30.0   # saturation inside the log-det


class ChannelNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class Reservoir(nn.Module):
    """Frozen random CNN. Stride 2 after the first stage, so `channels` sets the depth."""

    def __init__(self, width, seed, channels=(16, 32, 64, 64)):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            layers, inc = [], 3
            for i, c in enumerate(channels):
                layers += [nn.Conv2d(inc, c, 3, 1 if i == 0 else 2, 1), ChannelNorm(), nn.ELU()]
                inc = c
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(2), nn.Flatten(),
                                     nn.Linear(channels[-1] * 4, width))
        self.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


def ridge_map(h, lam=RIDGE):
    """(H^T H + lam I)^-1 H^T on standardised features, via QR. Shape [features, batch]."""
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])).double()
    aug = torch.cat([h, math.sqrt(lam) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)])
    q, r = torch.linalg.qr(aug, mode="reduced")
    return torch.linalg.solve_triangular(r, q[: h.shape[0]].T, upper=True)


def score(z, a, eta=ETA):
    z = z.double()
    w = a @ (z - z.mean(0))
    eye = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)
    return 0.5 * torch.linalg.slogdet(eye + eta * w.T @ w)[1] / math.log(2)


def heldout_score(z, h, eta=ETA, lam=RIDGE):
    """Cross-fitted score: only readout structure that two disjoint halves of the batch agree on counts.

    `score` measures W^T W for one readout W fitted on the whole batch. A ridge fit also finds weights for
    pure noise, so W^T W = (true structure) + (fitting noise), and the noise term is paid for in bits: unpredictable
    embeddings earn most of the score of predictable ones. Here the readout is fitted twice, W_a on the first
    half of the rows and W_b on the second, and W^T W is replaced by the symmetrised W_a^T W_b. The two fits
    share the true structure but their fitting noise is independent, so it averages out: noise earns ~0.

      S = 0.5 * sum_i sign(l_i) * log2(1 + eta * |l_i|),   l_i = eigenvalues of (W_a^T W_b + W_b^T W_a) / 2

    For l_i >= 0 this is the same log-det as `score`; the odd extension to l_i < 0 (the halves disagree in
    that direction) makes noise cancel in expectation instead of being clipped to a positive value.
    Rows must be ordered window-major with an even number of windows, so that the midpoint splits windows,
    not time steps of one window. Features are standardised on the whole batch (both fits use the same
    coordinates) and the ridge is scaled with the rows of each half (both fits estimate the full-batch W).
    """
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])).double()
    z = z.double()
    n, w = h.shape[0] // 2, []
    for hh, zz in ((h[:n], z[:n]), (h[n:], z[n:])):
        aug = torch.cat([hh, math.sqrt(lam * hh.shape[0] / h.shape[0]) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)])
        q, r = torch.linalg.qr(aug, mode="reduced")
        w.append(torch.linalg.solve_triangular(r, q[: hh.shape[0]].T, upper=True) @ (zz - zz.mean(0)))
    m = w[0].T @ w[1]
    ev = torch.linalg.eigvalsh(0.5 * (m + m.T))
    return 0.5 * (ev.sign() * torch.log1p(eta * ev.abs())).sum() / math.log(2)


def epiplexity(z, x, reservoir, heldout=False):
    """Score of embedding z (differentiable) against the frozen reservoir's view of images x."""
    with torch.no_grad():
        h = reservoir(x)
        a = None if heldout else ridge_map(h)
    return heldout_score(z, h) if heldout else score(z, a)

def dynamics_epiplexity(z, x_next, reservoir, heldout=False):
    return epiplexity(z, x_next, reservoir, heldout)


class LatentReservoir(nn.Module):
    """Frozen random MLP on (z_t, a_t): the bounded observer of the latent dynamics.

    z and a enter through separate random layers and are summed, so a low-dimensional action is not drowned
    out by a 192-d latent (each branch has the same output variance for unit-variance inputs).
    """

    def __init__(self, z_dim, a_dim, width, seed, hidden=512):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.fz, self.fa = nn.Linear(z_dim, hidden), nn.Linear(a_dim, hidden, bias=False)
            self.out = nn.Linear(hidden, width)
        self.requires_grad_(False)

    def forward(self, z, a):
        return self.out(F.elu(self.fz(z) + self.fa(a)))


def latent_dynamics_epiplexity(z, a, z_next, reservoir, detach_inputs=False, heldout=False):
    """Score of z_{t+1} against a ridge readout from frozen random features of (z_t, a_t).

    Unlike `epiplexity`, the features depend on the encoder (through z_t), so the ridge map is differentiable:
    the encoder is rewarded both for a predictable z_{t+1} and for a z_t whose random features predict it.
    detach_inputs=True stops the second path (gradient through the target only).
    """
    h = reservoir((z.detach() if detach_inputs else z).float(), a.float())
    return heldout_score(z_next, h) if heldout else score(z_next, ridge_map(h))
