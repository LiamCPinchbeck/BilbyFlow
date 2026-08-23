"""
bilbyflow.nn.embedding — the embedding: data in, context vector out.

Every embedding takes ALL the data streams — fd, td, psd, amp — and decides
what to do with them. Call it at whichever level you have data:

    emb.forward_raw(x)      raw x          -> context     (standardises first)
    emb.forward_std(x)      standardised x -> context     (no standardising)
    emb(x)                  == forward_raw(x)

    emb.standardise(x)      raw            -> standardised
    emb.unpack(x_std)       standardised   -> {fd, td, psd, amp}
    emb.embed(fd, td, psd)  blocks         -> context     <- you implement this

Custom embeddings subclass StrainEmbedding, declare the blocks they want, and
implement embed() with one argument per block:

    class MyEmbedding(StrainEmbedding):
        blocks = ("fd", "psd")           # no time domain
        output_dim = 512
        def embed(self, fd, psd):
            ...                          # (B, output_dim)

The built-ins below take fd, td AND psd: the strain streams go through their
conv branches, the PSD through an MLP, and the two are concatenated —
the production architecture, now explicit in embed() instead of hidden in a
wrapper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ["StrainEmbedding", "Conv1dEmbedding", "Conv1dResNetEmbedding",
           "ResNetEmbedding", "psd_mlp", "FDPSDEmbedding", "FiLMEmbedding", "MLPEmbedding"]


class StrainEmbedding(nn.Module):
    """Base class: standardisation + block unpacking. Subclasses implement
    embed(**blocks) -> (B, output_dim)."""

    blocks = ("fd", "td", "psd")     # what this embedding consumes
    output_dim = None                # context width (the flow needs it)

    def __init__(self, cfg, std=None, output_dim=None, blocks=None):
        super().__init__()
        self.cfg = cfg
        self.std = std
        self.output_dim = int(output_dim or self.output_dim
                              or cfg.get("embedding_output_dim", 128))

        g = _grid(cfg)
        self.n_masked, self.n_td = g["n_masked"], g["n_td"]
        self.psd_on = bool(cfg.get("psd_conditioning", False))
        self.amp_dim = 5 if cfg.get("amp_context", False) else 0

        # THE packed layout, matching data.canonical.build_x_full:
        #   [reH1, imH1, tdH1, reL1, imL1, tdL1] [|| psd] [|| amp]
        per_det = 2 * self.n_masked + self.n_td
        self.strain_dim = 2 * per_det
        self.psd_dim = 2 * self.n_masked if self.psd_on else 0
        self.sl = {"_total": self.strain_dim + self.psd_dim + self.amp_dim}

        want = [b for b in (blocks or self.blocks)
                if b != "psd" or self.psd_on]
        if not want:
            raise ValueError(f"{type(self).__name__}: no blocks to read")
        self._blocks = want

    def unpack(self, x):
        """Standardised x -> {fd, td, psd, amp}, filtered to the blocks this
        embedding declared. The packing is fixed (see __init__); `blocks`
        selects which views embed() receives, not what the dataset builds."""
        if x.shape[-1] != self.sl["_total"]:
            raise ValueError(
                f"{type(self).__name__}: x width {x.shape[-1]} != "
                f"{self.sl['_total']} (strain {self.strain_dim} + psd "
                f"{self.psd_dim} + amp {self.amp_dim})")
        nm, ntd, B = self.n_masked, self.n_td, x.shape[0]
        per = 2 * nm + ntd
        ch = []
        for o in (0, per):                       # H1 then L1
            ch += [x[:, o:o + nm], x[:, o + nm:o + 2 * nm],
                   x[:, o + 2 * nm:o + per]]
        re_h, im_h, td_h, re_l, im_l, td_l = ch

        avail = {"fd": torch.stack([re_h, im_h, re_l, im_l], dim=1),
                 "td": torch.stack([td_h, td_l], dim=1)}
        if self.psd_on:
            p = x[:, self.strain_dim:self.strain_dim + self.psd_dim]
            avail["psd"] = p.view(B, 2, nm)
        if self.amp_dim:
            avail["amp"] = x[:, -self.amp_dim:]
        out = {k: avail[k] for k in self._blocks if k in avail}
        if "amp" in avail:              # always passed through to forward_std,
            out["amp"] = avail["amp"]   # which appends it after embed()
        return out


    def standardise(self, x):
        """Raw x -> standardised x. Identity when no standardiser is set."""
        return x if self.std is None else self.std.normalise_x(x)


    @property
    def context_dim(self):
        """Width of forward()'s output — what the flow must be built for.
        Equals output_dim, plus the raw amp block when amp_context is on."""
        return self.output_dim + self.amp_dim
    

    def embed(self, **blocks):
        raise NotImplementedError

    # -------------------- entry points ---------------------------------
    def forward_std(self, x):
        """Already-standardised x -> context. The amp block (already
        z-scored) is appended after embed(), so embed() never has to accept
        it."""
        blocks = self.unpack(x)
        amp = blocks.pop("amp", None)
        out = self.embed(**blocks)
        return out if amp is None else torch.cat([out, amp], dim=1)


    def forward_raw(self, x):
        """Raw x -> context (standardises first)."""
        return self.forward_std(self.standardise(x))


    def forward(self, x):
        return self.forward_raw(x)


def _grid(cfg):
    duration, sr = float(cfg["duration"]), int(cfg["sampling_frequency"])
    n_fd_full = int(duration * sr / 2) + 1
    freqs = np.arange(n_fd_full) / duration
    return dict(n_masked=int((freqs >= float(cfg["f_min"])).sum()),
                n_td=2 * (n_fd_full - 1))


# -------------------- shared pieces -----------------------

def psd_mlp(in_dim, cfg):
    """The PSD encoder: LayerNorm+ELU MLP, psd_encoder_hidden -> psd_out."""
    dims = ([in_dim] + list(cfg.get("psd_encoder_hidden", [512, 256, 128]))
            + [int(cfg.get("psd_encoder_out", 64))])
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.ELU()]
    return nn.Sequential(*layers)


def _make_resnet(rtype, in_channels, output_dim):
    if rtype not in ("resnet18", "resnet34", "resnet50"):
        raise ValueError(f"Unknown resnet backbone: {rtype}")
    net = getattr(models, rtype)(weights=None)
    net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, output_dim)
    return net


class _ConvStem(nn.Module):
    """Strided 1-D conv stack -> (B, C, L') feature map."""

    def __init__(self, c_in, channels, k):
        super().__init__()
        layers = []
        for c_out in channels:
            layers += [nn.Conv1d(c_in, c_out, k, stride=2, padding=k // 2,
                                 bias=False), nn.BatchNorm1d(c_out), nn.ELU()]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x):
        return self.net(x)


class _Branch(nn.Module):
    """1-D stem -> fold features to a near-square grid -> 2-D ResNet."""

    def __init__(self, c_in, channels, k, stem_out, backbone, out_dim, ref_len):
        super().__init__()
        stem_ch = list(channels[:-1]) + [stem_out]
        self.stem = _ConvStem(c_in, stem_ch, k)
        length = ref_len
        for _ in stem_ch:
            length = (length + 2 * (k // 2) - k) // 2 + 1
        self.w = int(np.ceil(np.sqrt(length)))
        self.h = int(np.ceil(length / self.w))
        self.resnet = _make_resnet(backbone, stem_out, out_dim)

    def forward(self, x):
        f = self.stem(x)
        size = self.h * self.w
        f = F.pad(f, (0, max(0, size - f.shape[2])))[:, :, :size]
        return self.resnet(f.view(f.shape[0], f.shape[1], self.h, self.w))


def _fit(t, size):
    """Pad or truncate the last axis to `size`."""
    return F.pad(t, (0, max(0, size - t.shape[-1])))[..., :size]


# -------------------- built-ins: all take fd, td AND psd ---------------------------------

class Conv1dEmbedding(StrainEmbedding):
    """Two 1-D conv stems (FD, TD) + PSD MLP -> concat -> MLP head."""

    blocks = ("fd", "td", "psd")

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)
        ch = list(cfg.get("conv1d_channels", [32, 64, 128, 128]))
        k = int(cfg.get("conv1d_kernel", 7))
        self.fd_stem = _ConvStem(4, ch, k)
        self.td_stem = _ConvStem(2, ch, k)
        self.pool = nn.AdaptiveAvgPool1d(1)

        feat = self.fd_stem.out_ch + self.td_stem.out_ch
        if "psd" in self._blocks:
            self.psd_encoder = psd_mlp(2 * self.n_masked, cfg)
            feat += int(cfg.get("psd_encoder_out", 64))
        self.head = nn.Sequential(
            nn.Linear(feat, 2 * self.output_dim), nn.ELU(),
            nn.Dropout(float(cfg.get("conv1d_dropout", 0.1))),
            nn.Linear(2 * self.output_dim, self.output_dim))

    def embed(self, fd, td, psd=None):
        feats = [self.pool(self.fd_stem(fd)).squeeze(-1),
                 self.pool(self.td_stem(td)).squeeze(-1)]
        if psd is not None:
            feats.append(self.psd_encoder(psd.flatten(1)))
        return self.head(torch.cat(feats, dim=1))


class Conv1dResNetEmbedding(StrainEmbedding):
    """FD and TD branches (1-D stem -> 2-D ResNet) + PSD MLP -> MLP head."""

    blocks = ("fd", "td", "psd")

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)
        args = (list(cfg.get("conv1d_channels", [32, 64, 128, 128])),
                int(cfg.get("conv1d_kernel", 7)),
                int(cfg.get("conv1d_resnet_stem_out", 64)),
                cfg.get("conv1d_resnet_backbone", "resnet18"),
                self.output_dim)
        self.fd_branch = _Branch(4, *args, self.n_masked)
        self.td_branch = _Branch(2, *args, self.n_td)

        feat = 2 * self.output_dim
        if "psd" in self._blocks:
            self.psd_encoder = psd_mlp(2 * self.n_masked, cfg)
            feat += int(cfg.get("psd_encoder_out", 64))
        self.head = nn.Sequential(
            nn.Linear(feat, 2 * self.output_dim), nn.ELU(),
            nn.Dropout(float(cfg.get("conv1d_dropout", 0.1))),
            nn.Linear(2 * self.output_dim, self.output_dim))

    def embed(self, fd, td, psd=None):
        feats = [self.fd_branch(fd), self.td_branch(td)]
        if psd is not None:
            feats.append(self.psd_encoder(psd.flatten(1)))
        return self.head(torch.cat(feats, dim=1))



class ResNetEmbedding(StrainEmbedding):
    """Fold FD+TD into a 6-channel 2-D grid, one ResNet, + PSD MLP.

    NOTE the channel order is [fd(4), td(2)], not the historical interleave
    [reH1, imH1, tdH1, reL1, imL1, tdL1] — old checkpoints do not load here."""

    blocks = ("fd", "td", "psd")

    def __init__(self, cfg, std=None, output_dim=None, backbone="resnet34"):
        super().__init__(cfg, std, output_dim)
        L = max(self.n_masked, self.n_td)
        self.img_w = int(np.ceil(np.sqrt(L)))
        self.img_h = int(np.ceil(L / self.img_w))
        self.resnet = _make_resnet(backbone, 6, self.output_dim)

        if "psd" in self._blocks:
            self.psd_encoder = psd_mlp(2 * self.n_masked, cfg)
            self.head = nn.Linear(
                self.output_dim + int(cfg.get("psd_encoder_out", 64)),
                self.output_dim)

    def embed(self, fd, td, psd=None):
        size = self.img_h * self.img_w
        chans = [fd[:, 0], fd[:, 1], td[:, 0],      # reH1, imH1, tdH1
                 fd[:, 2], fd[:, 3], td[:, 1]]      # reL1, imL1, tdL1
        x = torch.stack([_fit(c.unsqueeze(1), size).squeeze(1) for c in chans],
                        dim=1)
        out = self.resnet(x.view(x.shape[0], 6, self.img_h, self.img_w))
        if psd is None:
            return out
        return self.head(torch.cat(
            [out, self.psd_encoder(psd.flatten(1))], dim=1))


class FDPSDEmbedding(StrainEmbedding):
    """FD + PSD as ONE 6-channel stream, no time domain.

    fd (4, n_masked) and psd (2, n_masked) live on the same frequency axis, so
    concatenating them as channels is exact per-bin fusion: the conv can learn
    "distrust this bin, there's a line here", which late fusion after global
    pooling cannot. The TD channels are dropped because they are a
    deterministic irfft of the FD ones — no new information, only a different
    inductive bias. This is therefore both the PSD-fusion experiment and the
    TD ablation."""

    blocks = ("fd", "psd")

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)
        if "psd" not in self._blocks:
            raise ValueError("FDPSDEmbedding needs psd_conditioning: true")
        self.branch = _Branch(
            6,                                              # 4 FD + 2 PSD
            list(cfg.get("conv1d_channels", [32, 64, 128, 128])),
            int(cfg.get("conv1d_kernel", 7)),
            int(cfg.get("conv1d_resnet_stem_out", 64)),
            cfg.get("conv1d_resnet_backbone", "resnet18"),
            self.output_dim, self.n_masked)
        self.head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim), nn.ELU(),
            nn.Dropout(float(cfg.get("conv1d_dropout", 0.1))),
            nn.Linear(self.output_dim, self.output_dim))

    def embed(self, fd, psd):
        return self.head(self.branch(torch.cat([fd, psd], dim=1)))


class FiLMEmbedding(StrainEmbedding):
    """FD/TD conv stems whose feature maps are MODULATED by the PSD.

    The PSD encoder emits per-channel (gamma, beta) and the stems' features
    become gamma*F + beta (Perez et al., AAAI 2018, arXiv:1709.07871).
    Initialised to gamma=1, beta=0, so training starts exactly at the
    unconditioned model — strictly more expressive than concatenation, and
    stable from step one."""

    blocks = ("fd", "td", "psd")

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)
        ch = list(cfg.get("conv1d_channels", [32, 64, 128, 128]))
        k = int(cfg.get("conv1d_kernel", 7))
        self.fd_stem = _ConvStem(4, ch, k)
        self.td_stem = _ConvStem(2, ch, k)
        self.pool = nn.AdaptiveAvgPool1d(1)

        n_ch = self.fd_stem.out_ch + self.td_stem.out_ch
        self.psd_encoder = psd_mlp(2 * self.n_masked, cfg)
        self.film = nn.Linear(int(cfg.get("psd_encoder_out", 64)), 2 * n_ch)
        nn.init.zeros_(self.film.weight)          # identity at init:
        nn.init.zeros_(self.film.bias)            # gamma = 1, beta = 0
        self.n_fd_ch = self.fd_stem.out_ch

        self.head = nn.Sequential(
            nn.Linear(n_ch, 2 * self.output_dim), nn.ELU(),
            nn.Dropout(float(cfg.get("conv1d_dropout", 0.1))),
            nn.Linear(2 * self.output_dim, self.output_dim))

    def embed(self, fd, td, psd=None):
        f, t = self.fd_stem(fd), self.td_stem(td)
        if psd is not None:
            gb = self.film(self.psd_encoder(psd.flatten(1)))
            gamma, beta = gb.chunk(2, dim=-1)
            gamma, beta = 1.0 + gamma, beta                # identity at init
            gf, gt = gamma[:, :self.n_fd_ch], gamma[:, self.n_fd_ch:]
            bf, bt = beta[:, :self.n_fd_ch], beta[:, self.n_fd_ch:]
            f = gf.unsqueeze(-1) * f + bf.unsqueeze(-1)
            t = gt.unsqueeze(-1) * t + bt.unsqueeze(-1)
        return self.head(torch.cat([self.pool(f).squeeze(-1),
                                    self.pool(t).squeeze(-1)], dim=1))


class MLPEmbedding(StrainEmbedding):
    """Flatten everything into one MLP. Weak (no locality at all) but it is
    the shortest possible example of the contract, and a useful control."""

    blocks = ("fd", "td", "psd")

    def __init__(self, cfg, std=None, output_dim=None):
        super().__init__(cfg, std, output_dim)
        # widths of the blocks embed() actually receives; amp is appended
        # after embed() by forward_std, so it is excluded here
        widths = {"fd": 4 * self.n_masked, "td": 2 * self.n_td,
                  "psd": self.psd_dim}
        in_dim = sum(widths[k] for k in self._blocks if k in widths)
        dims = [in_dim] + list(cfg.get("embedding_layers", [512, 512, 256]))
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ELU()]
        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.net = nn.Sequential(*layers)

    def embed(self, fd, td, psd=None):
        parts = [fd.flatten(1), td.flatten(1)]
        if psd is not None:
            parts.append(psd.flatten(1))
        return self.net(torch.cat(parts, dim=1))