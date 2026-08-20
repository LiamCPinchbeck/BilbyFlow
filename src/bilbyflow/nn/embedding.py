"""
bilbyflow.nn.embedding — strain embedding networks.

Three embeddings map the packed strain vector x to a
context vector for the flow. All share the same inputs (the
6-channel FD/TD packing of ``_split_slices``) and the same output dim,
so ``cfg["embedding_type"]`` selects between them.

    from bilbyflow.nn.embedding import build_resnet_embedding
    embedding = build_resnet_embedding(input_dim, cfg)

Selected by cfg["embedding_type"]:
    "conv1d"                           -> Conv1dEmbedding       (light, recommended)
    "conv1d_resnet"                    -> Conv1dResNetEmbedding (1-D stem -> ResNet)
    "resnet18"/"resnet34"/"resnet50"   -> ResNetEmbedding       (legacy 2-D reshape)

Trade-off:
  * conv1d        : adjacency-preserving, ~0.4M params  -> least memorisation
  * conv1d_resnet : adjacency-preserving 1-D stem feeds a full ResNet
                    -> keeps ResNet expressivity, keeps most of its params
                    -> seems to be the easiest to handle/train/get good results from
  * resnet*       : raw 1-D vector folded into a 2-D grid (adjacency broken)

Config keys (optional, with defaults):
    embedding_type: conv1d
    embedding_output_dim: 128

    conv1d / conv1d_resnet stem:
        conv1d_channels: [32, 64, 128, 128]
        conv1d_kernel: 7
        conv1d_dropout: 0.1

    conv1d_resnet extra:
        conv1d_resnet_stem_out: 64
        conv1d_resnet_backbone: resnet18

NOTE ON CHECKPOINT COMPATIBILITY: the state-dict keys are
"{fd,td}_branch.*" for Conv1dResNetEmbedding and 
"{fd,td}_stem.*" for Conv1dEmbedding. 
The reweighting checkpoint loader remaps
".fd_branch."<->".fd_stem." to bridge older checkpoints
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

__all__ = [
    "build_resnet_embedding",
    "Conv1dEmbedding",
    "Conv1dResNetEmbedding",
    "ResNetEmbedding",
]


# ── grid helpers ─────────────────────────────────────────────────────────────

def _channel_sizes(cfg):
    """(n_fd_masked, n_td) from the config grid — the per-channel lengths."""
    duration = float(cfg["duration"])
    sr = int(cfg["sampling_frequency"])
    f_min = float(cfg["f_min"])
    n_fd_full = int(duration * sr / 2) + 1
    n_td = 2 * (n_fd_full - 1)
    freq_array = np.arange(n_fd_full) / duration
    n_fd_masked = int(np.sum(freq_array >= f_min))
    return n_fd_masked, n_td


def _split_slices(n_fd_masked, n_td):
    """Byte-offset ranges of each of the 6 channels within the packed x."""
    per_det = 2 * n_fd_masked + n_td
    return dict(
        re_h1=(0, n_fd_masked),
        im_h1=(n_fd_masked, 2 * n_fd_masked),
        td_h1=(2 * n_fd_masked, 2 * n_fd_masked + n_td),
        re_l1=(per_det, per_det + n_fd_masked),
        im_l1=(per_det + n_fd_masked, per_det + 2 * n_fd_masked),
        td_l1=(per_det + 2 * n_fd_masked, per_det + 2 * n_fd_masked + n_td),
    )


def _fd_td_inputs(x, sl):
    """Pack x into FD (4-channel: H1/L1 re/im) and TD (2-channel) tensors."""
    def seg(key):
        a, b = sl[key]
        return x[:, a:b]
    fd = torch.stack([seg("re_h1"), seg("im_h1"), seg("re_l1"), seg("im_l1")], dim=1)
    td = torch.stack([seg("td_h1"), seg("td_l1")], dim=1)
    return fd, td


def _make_resnet(resnet_type, in_channels, output_dim):
    """A torchvision ResNet with the stem conv and head re-sized for our data."""
    if resnet_type == "resnet18":
        net = models.resnet18(weights=None)
    elif resnet_type == "resnet34":
        net = models.resnet34(weights=None)
    elif resnet_type == "resnet50":
        net = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown resnet backbone: {resnet_type}")
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, output_dim)
    return net


# ── shared 1-D pieces ────────────────────────────────────────────────────────

def _conv_block(c_in, c_out, k):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, stride=2, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ELU(),
    )


class _ConvStem(nn.Module):
    """Strided 1-D conv stack. Returns the (B, C, L') feature map (no pooling)."""
    def __init__(self, c_in, channels, kernel):
        super().__init__()
        layers, c = [], c_in
        for c_out in channels:
            layers.append(_conv_block(c, c_out, kernel))
            c = c_out
        self.net = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x):
        return self.net(x)


# ── entry point ──────────────────────────────────────────────────────────────

def build_resnet_embedding(input_dim, cfg):
    """Dispatch on embedding_type. Name kept for drop-in compatibility.

    input_dim is accepted for signature compatibility; the per-channel
    lengths are derived from the config grid via _channel_sizes.
    """
    n_fd_masked, n_td = _channel_sizes(cfg)
    output_dim = int(cfg.get("embedding_output_dim", 128))
    etype = cfg.get("embedding_type", "conv1d")

    if etype == "conv1d":
        return Conv1dEmbedding(
            n_fd_masked=n_fd_masked, n_td=n_td, output_dim=output_dim,
            channels=list(cfg.get("conv1d_channels", [32, 64, 128, 128])),
            kernel=int(cfg.get("conv1d_kernel", 7)),
            dropout=float(cfg.get("conv1d_dropout", 0.1)),
        )
    if etype == "conv1d_resnet":
        return Conv1dResNetEmbedding(
            n_fd_masked=n_fd_masked, n_td=n_td, output_dim=output_dim,
            channels=list(cfg.get("conv1d_channels", [32, 64, 128, 128])),
            kernel=int(cfg.get("conv1d_kernel", 7)),
            dropout=float(cfg.get("conv1d_dropout", 0.1)),
            stem_out=int(cfg.get("conv1d_resnet_stem_out", 64)),
            backbone=cfg.get("conv1d_resnet_backbone", "resnet18"),
        )
    if etype in ("resnet18", "resnet34", "resnet50"):
        return ResNetEmbedding(
            n_fd_masked=n_fd_masked, n_td=n_td,
            output_dim=output_dim, resnet_type=etype,
        )
    raise ValueError(f"Unknown embedding_type: {etype}")


# ── option 1: light 1-D conv embedding ──────────────────────────────────────

class Conv1dEmbedding(nn.Module):
    """Two 1-D conv stems (FD, TD) -> global pool -> MLP head. Adjacency-preserving."""
    def __init__(self, n_fd_masked, n_td, output_dim=128,
                 channels=(32, 64, 128, 128), kernel=7, dropout=0.1):
        super().__init__()
        self.sl = _split_slices(n_fd_masked, n_td)
        self.fd_stem = _ConvStem(4, list(channels), kernel)
        self.td_stem = _ConvStem(2, list(channels), kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        feat = self.fd_stem.out_ch + self.td_stem.out_ch
        self.head = nn.Sequential(
            nn.Linear(feat, 2 * output_dim), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(2 * output_dim, output_dim),
        )

    def forward(self, x):
        fd, td = _fd_td_inputs(x, self.sl)
        f = self.pool(self.fd_stem(fd)).squeeze(-1)
        t = self.pool(self.td_stem(td)).squeeze(-1)
        return self.head(torch.cat([f, t], dim=1))


# ── option 2: 1-D stem -> 2-D ResNet (expressivity + correct adjacency) ─────

class _Reshape1dTo2d(nn.Module):
    """Pad a (B, C, L) feature map and fold L into a near-square (B, C, h, w)."""
    def __init__(self, length):
        super().__init__()
        self.w = int(np.ceil(np.sqrt(length)))
        self.h = int(np.ceil(length / self.w))
        self.size = self.h * self.w

    def forward(self, x):                       # (B, C, L)
        b, c, l = x.shape
        if l < self.size:
            x = torch.cat([x, x.new_zeros(b, c, self.size - l)], dim=2)
        else:
            x = x[:, :, :self.size]
        return x.view(b, c, self.h, self.w)


class _Conv1dResNetBranch(nn.Module):
    """1-D conv stem -> reshape features to a grid -> per-branch ResNet."""
    def __init__(self, c_in, channels, kernel, stem_out, backbone, output_dim, ref_len):
        super().__init__()
        # stem widths end at stem_out (channels handed to the 2-D conv)
        stem_channels = list(channels[:-1]) + [stem_out]
        self.stem = _ConvStem(c_in, stem_channels, kernel)
        # length after len-1 stride-2 layers (one per stem layer)
        l_out = ref_len
        for _ in stem_channels:
            l_out = (l_out + 2 * (kernel // 2) - kernel) // 2 + 1
        self.reshape = _Reshape1dTo2d(l_out)
        self.resnet = _make_resnet(backbone, in_channels=stem_out, output_dim=output_dim)

    def forward(self, x):
        return self.resnet(self.reshape(self.stem(x)))


class Conv1dResNetEmbedding(nn.Module):
    """1-D stem per branch feeding a 2-D ResNet; concatenated -> MLP head."""
    def __init__(self, n_fd_masked, n_td, output_dim=128,
                 channels=(32, 64, 128, 128), kernel=7, dropout=0.1,
                 stem_out=64, backbone="resnet18"):
        super().__init__()
        self.sl = _split_slices(n_fd_masked, n_td)
        branch_dim = output_dim
        self.fd_branch = _Conv1dResNetBranch(4, channels, kernel, stem_out,
                                             backbone, branch_dim, n_fd_masked)
        self.td_branch = _Conv1dResNetBranch(2, channels, kernel, stem_out,
                                             backbone, branch_dim, n_td)
        self.head = nn.Sequential(
            nn.Linear(2 * branch_dim, 2 * output_dim), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(2 * output_dim, output_dim),
        )

    def forward(self, x):
        fd, td = _fd_td_inputs(x, self.sl)
        return self.head(torch.cat([self.fd_branch(fd), self.td_branch(td)], dim=1))


# ── option 3: legacy 2-D ResNet (raw reshape, adjacency broken) ─────────────

class ResNetEmbedding(nn.Module):
    """Fold the raw packed x into a 6-channel 2-D grid and run one ResNet.

    Kept for loading older checkpoints; the raw reshape does not preserve
    frequency/time adjacency, so prefer conv1d or conv1d_resnet for new runs.
    """
    def __init__(self, n_fd_masked, n_td, output_dim=128, resnet_type="resnet34"):
        super().__init__()
        self.n_fd_masked = n_fd_masked
        self.n_td = n_td
        self.n_channels = 6
        self.channel_len = max(n_fd_masked, n_td)
        self.img_w = int(np.ceil(np.sqrt(self.channel_len)))
        self.img_h = int(np.ceil(self.channel_len / self.img_w))
        self.img_size = self.img_h * self.img_w
        self.resnet = _make_resnet(resnet_type, in_channels=self.n_channels,
                                   output_dim=output_dim)
        self.splits = list(_split_slices(n_fd_masked, n_td).values())

    def forward(self, x):
        batch_size = x.shape[0]
        channels = []
        for start, end in self.splits:
            ch = x[:, start:end]
            if ch.shape[1] < self.img_size:
                pad = torch.zeros(batch_size, self.img_size - ch.shape[1],
                                  device=x.device, dtype=x.dtype)
                ch = torch.cat([ch, pad], dim=1)
            else:
                ch = ch[:, :self.img_size]
            channels.append(ch)
        x_multi = torch.stack(channels, dim=1)
        x_2d = x_multi.view(batch_size, self.n_channels, self.img_h, self.img_w)
        return self.resnet(x_2d)