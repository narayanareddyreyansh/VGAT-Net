"""
Core building blocks of VGAT-Net.

Design note
-----------
The Multi-Scale Vessel Topology Encoder extracts multi-scale vessel features at
full resolution and then produces a hierarchy of down-sampled features (skips +
bottleneck). The Transformer Context Aggregator and Vessel-Guided FAZ Attention
operate at the bottleneck resolution (H/16), which keeps self/cross-attention
tractable, and the two decoders restore full resolution through attention-guided
skip connections. Feature width `C` (default 64) is kept constant across the
attention stack so it matches the F in R^{H x W x C} notation of the paper; the
width is configurable in configs/*.yaml.

Component -> paper section:
  MultiScaleVesselTopologyEncoder  -> Sec. 3.2 (Eqs. 3-5)
  VesselContinuityAttentionModule  -> Sec. 3.3 (Eqs. 6-8)
  TransformerContextAggregator     -> Sec. 3.4 (Eqs. 9-12)
  VesselGuidedFAZAttention         -> Sec. 3.5 (Eqs. 13-14)
  CrossTaskInteractionBlock        -> Sec. 3.6 (Eq. 15)
  SegmentationDecoder (x2)         -> Sec. 3.7 (Eqs. 16-17)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_ch, out_ch, k=3, dilation=1, act=True):
    pad = dilation * (k // 2)
    layers = [nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Down(nn.Module):
    """2x downsampling block: max-pool + double conv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_act(in_ch, out_ch, 3),
            conv_bn_act(out_ch, out_ch, 3),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------------------- #
# 3.2  Multi-Scale Vessel Topology Encoder (MVTE)
# --------------------------------------------------------------------------- #
class MultiScaleVesselTopologyEncoder(nn.Module):
    """Multi-scale stem (3x3, 5x5, dilated-3x3) + hierarchical downsampling.

    Returns (bottleneck, skips) where skips are high->low resolution.
    """

    def __init__(self, in_ch=1, base=64, depth=4):
        super().__init__()
        self.stem = conv_bn_act(in_ch, base, 3)
        self.local = conv_bn_act(base, base, 3)                 # F3
        self.medium = conv_bn_act(base, base, 5)                # F5
        self.longrange = conv_bn_act(base, base, 3, dilation=2)  # Fd
        self.fuse = conv_bn_act(base * 3, base, 1)              # 1x1 fusion (F_enc)
        self.downs = nn.ModuleList([Down(base, base) for _ in range(depth)])

    def forward(self, x):
        s = self.stem(x)
        f3, f5, fd = self.local(s), self.medium(s), self.longrange(s)
        f_enc = self.fuse(torch.cat([f3, f5, fd], dim=1))       # full-res skip
        skips = [f_enc]
        h = f_enc
        for down in self.downs:
            h = down(h)
            skips.append(h)
        bottleneck = skips.pop()          # lowest-resolution feature
        return bottleneck, skips          # skips: high->low res


# --------------------------------------------------------------------------- #
# 3.3  Vessel Continuity Attention Module (VCAM)
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        hidden = max(channels // r, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden, channels, bias=False))

    def forward(self, x):
        b, c, _, _ = x.shape
        z = x.mean(dim=(2, 3))                  # global average pooling (Eq. 6)
        a_c = torch.sigmoid(self.mlp(z)).view(b, c, 1, 1)
        return x * a_c                          # Eq. 7


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        a_s = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))  # Eq. 8
        return x * a_s


class VesselContinuityAttentionModule(nn.Module):
    def __init__(self, channels, r=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, r)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))   # F_v = A_s ⊙ (A_c ⊙ F_enc)


# --------------------------------------------------------------------------- #
# 3.4  Transformer Context Aggregator (TCA)
# --------------------------------------------------------------------------- #
def sinusoidal_position_encoding(n_tokens, dim, device):
    pe = torch.zeros(n_tokens, dim, device=device)
    pos = torch.arange(0, n_tokens, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TransformerContextAggregator(nn.Module):
    def __init__(self, channels, embed_dim=512, depth=6, heads=8,
                 mlp_ratio=4.0, patch_size=2, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.unproj = nn.ConvTranspose2d(embed_dim, channels,
                                         kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.proj(x)                                # patch embedding
        _, d, hp, wp = t.shape
        tokens = t.flatten(2).transpose(1, 2)           # (B, N, D)
        tokens = tokens + sinusoidal_position_encoding(tokens.size(1), d, x.device)  # Z0=E+PE
        tokens = self.encoder(tokens)                   # MHSA + FFN + LN (Eqs. 10-12)
        t = tokens.transpose(1, 2).reshape(b, d, hp, wp)
        out = self.unproj(t)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


# --------------------------------------------------------------------------- #
# 3.5  Vessel-Guided FAZ Attention Module (VGFA)
# --------------------------------------------------------------------------- #
class VesselGuidedFAZAttention(nn.Module):
    """Cross-attention: FAZ features (query) attend to vessel features (key/value)."""

    def __init__(self, channels, heads=8):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.conv_v = nn.Conv2d(channels, channels, 1)
        self.conv_f = nn.Conv2d(channels, channels, 1)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))        # γ initialized to 0 (Eq. 14)

    def _split(self, x):
        b, c, h, w = x.shape
        return x.view(b, self.heads, c // self.heads, h * w).permute(0, 1, 3, 2)

    def forward(self, f_t):
        b, c, h, w = f_t.shape
        f_v = self.conv_v(f_t)
        f_f = self.conv_f(f_t)
        q, k, v = self._split(self.q(f_f)), self._split(self.k(f_v)), self._split(self.v(f_v))
        scale = (c // self.heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)   # Eq. 13
        f_att = torch.matmul(attn, v)                                                # A_vf V
        f_att = f_att.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        f_att = self.out(f_att)
        f_faz = f_f + self.gamma * f_att                                             # Eq. 14
        return f_v, f_faz


# --------------------------------------------------------------------------- #
# 3.6  Cross-Task Interaction Block (CTIB)
# --------------------------------------------------------------------------- #
class CrossTaskInteractionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, f_v, f_faz):
        return self.fuse(torch.cat([f_v, f_faz], dim=1))    # Eq. 15


# --------------------------------------------------------------------------- #
# 3.7  Dual Segmentation Decoders
# --------------------------------------------------------------------------- #
class AttentionGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.wg = nn.Conv2d(ch, ch, 1)
        self.wx = nn.Conv2d(ch, ch, 1)
        self.psi = nn.Conv2d(ch, 1, 1)

    def forward(self, g, x):
        a = torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))
        return x * a


class UpStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.gate = AttentionGate(out_ch)
        self.conv = nn.Sequential(conv_bn_act(out_ch * 2, out_ch, 3),
                                  conv_bn_act(out_ch, out_ch, 3))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        s = self.gate(x, skip)                              # attention-guided skip (Eq. 16)
        return self.conv(torch.cat([x, s], dim=1))


class SegmentationDecoder(nn.Module):
    """Upsamples the fused bottleneck to full resolution using encoder skips."""

    def __init__(self, channels, num_stages=4):
        super().__init__()
        self.stages = nn.ModuleList([UpStage(channels, channels) for _ in range(num_stages)])
        self.head = nn.Conv2d(channels, 1, 1)               # 1x1 -> logits (Eq. 17)

    def forward(self, x, skips, out_size):
        # skips are high->low res; consume low->high as we upsample
        for stage, skip in zip(self.stages, reversed(skips)):
            x = stage(x, skip)
        logits = self.head(x)
        if logits.shape[-2:] != tuple(out_size):
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits
