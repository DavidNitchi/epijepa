"""CPU smoke test for train.py's regulariser plumbing: no dataset, no GPU, a few seconds.

  python worldmodel/smoke_test.py

Builds the real LeWM (epi arm) from config/train, feeds it a tiny random batch, and checks for every
loss.epi.target that the score rows are paired within a window, the score is finite and positive, and the
gradient reaches the frames it should. scripts/setup_rorqual_env.sh runs this after installing the env.
"""
import sys
from pathlib import Path

import hydra
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from epiplexity import (LatentReservoir, Reservoir, dynamics_epiplexity, epiplexity,  # noqa: E402
                        heldout_score, latent_dynamics_epiplexity)

torch.manual_seed(0)
base = ["data=tworoom", "model=epi", "loss.reg=epi"]
cfg = train.load_cfg(base)
assert (cfg.loss.epi.target, cfg.loss.epi.detach_inputs, cfg.loss.epi.heldout) == ("frame", False, False), "defaults must stay EpiWM"
assert train.load_cfg(base + ["loss.epi.target=latent"]).loss.epi.target == "latent"

# pairing, with ids 10*window + time written into every tensor
B, T, D = 5, 4, 3
ids = (torch.arange(B)[:, None] * 10 + torch.arange(T)[None]).float()
emb, act = ids[..., None].expand(B, T, D), ids[..., None].expand(B, T, 2)
pix = ids[..., None, None, None].expand(B, T, 3, 8, 8)
z, x = train.reg_rows(emb, pix, act, "frame")
assert torch.equal(z, emb.reshape(-1, D)) and torch.equal(x, pix.reshape(-1, 3, 8, 8))
z, x = train.reg_rows(emb, pix, act, "next")
assert torch.equal(x[:, 0, 0, 0], z[:, 0] + 1)
z, a, zn = train.reg_rows(emb, pix, act, "latent")
assert z.shape == (B * (T - 1), D) and torch.equal(a[:, 0], z[:, 0]) and torch.equal(zn[:, 0], z[:, 0] + 1)
assert torch.equal(train.reg_rows(emb, pix, act, "latent_delta")[2], torch.ones(B * (T - 1), D))
print("pairing ok")

# the real model on a tiny random batch
A = cfg.data.dataset.frameskip * 2
cfg.model.action_encoder.input_dim = A
model = hydra.utils.instantiate(cfg.model).train()
cnn = Reservoir(cfg.loss.epi.width, cfg.loss.epi.reservoir_seed).eval()
lat = LatentReservoir(cfg.model.projector.output_dim, A, cfg.loss.epi.width, cfg.loss.epi.reservoir_seed).eval()
B, T, H, P = 8, cfg.history_size + cfg.num_preds, cfg.history_size, cfg.num_preds
batch = dict(pixels=torch.randn(B, T, 3, cfg.img_size, cfg.img_size), action=torch.randn(B, T, A))
out = model.encode(batch)
emb = out["emb"].float()
l_pred = (model.predict(emb[:, :H], out["act_emb"][:, :H]).float() - emb[:, P:]).pow(2).mean()
for target, detach in (("frame", False), ("next", False), ("latent", False), ("latent", True), ("latent_delta", False)):
    rows = train.reg_rows(emb, batch["pixels"], batch["action"], target)
    if target.startswith("latent"):
        s = latent_dynamics_epiplexity(*rows, lat, detach)
    else:
        s = (dynamics_epiplexity if target == "next" else epiplexity)(*rows, cnn)
    g = torch.autograd.grad(s, emb, retain_graph=True)[0].abs().sum(-1).mean(0)  # gradient per frame index
    assert torch.isfinite(s) and s > 0 and torch.isfinite(g).all(), target
    want = dict(frame=[1, 1, 1, 1], next=[1, 1, 1, 0], latent=[0 if detach else 1, 1, 1, 1], latent_delta=[1, 1, 1, 1])[target]
    assert [int(v > 0) for v in g] == want, (target, detach, g)
    if target.startswith("latent"):
        sh = latent_dynamics_epiplexity(*rows, lat, detach, heldout=True)
    else:
        sh = (dynamics_epiplexity if target == "next" else epiplexity)(*rows, cnn, heldout=True)
    gh = torch.autograd.grad(sh, emb, retain_graph=True)[0]
    assert torch.isfinite(sh) and torch.isfinite(gh).all() and gh.abs().sum() > 0, (target, "heldout")
    print(f"target={target:12s} detach_inputs={detach!s:5s} rows={rows[0].shape[0]:3d} S={float(s):7.2f} "
          f"held-out S={float(sh):7.2f} grad reaches frames {want}")

# held-out scoring at the training row count: noise must earn ~nothing, structure must not
g = torch.Generator().manual_seed(0)
h = torch.randn(384, 192, generator=g)
noise, structure = torch.randn(384, 192, generator=g), h @ torch.linalg.qr(torch.randn(192, 192, generator=g))[0]
s_noise, s_struct = float(heldout_score(noise, h)), float(heldout_score(structure, h))
assert abs(s_noise) < 0.05 * s_struct, (s_noise, s_struct)
print(f"held-out score: noise {s_noise:.1f} vs readable structure {s_struct:.1f}")
(l_pred - cfg.loss.epi.weight * s / float(s.detach())).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print(f"smoke test passed | torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
