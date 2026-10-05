"""Baseline model registry for fair comparison (Sec. 4.2).

Native, runnable implementations are provided for U-Net and Attention U-Net.
For the Transformer baselines (TransUNet, Swin-UNet, MISSFormer) and the recent
OCTA-specific methods, we use the authors' official public code; the exact
sources are recorded in configs/baseline_*.yaml under `impl_source`. To plug one
in, implement `build(cfg)` returning an nn.Module whose forward returns a dict
{"vessel": logits, "faz": logits} and register it below.

Baseline implementation sources
-------------------------------
  TransUNet      : https://github.com/Beckschen/TransUNet
  Swin-UNet      : https://github.com/HuCaoFighting/Swin-Unet
  MISSFormer     : https://github.com/ZhifangDeng/MISSFormer
  S2A-Net  [12]  : per paper (Biomed. Signal Process. Control, 2025)
  DGNet    [13]  : per paper (Biomed. Signal Process. Control, 2025)
  HV-OCTAMamba[18]: authors' public release (URL recorded in configs/baseline_hv_octamamba.yaml when added)
  DDNet    [25]  : per paper (J. Supercomputing, 2026)
  Dense-PMSFNet[19]: per paper (IEEE LATAM, 2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cba(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class _UNet(nn.Module):
    """Two-head U-Net (shared encoder, two 1x1 output heads)."""

    def __init__(self, in_ch=1, base=32, attention=False):
        super().__init__()
        self.attention = attention
        self.e1, self.e2, self.e3, self.e4 = (_cba(in_ch, base), _cba(base, base * 2),
                                              _cba(base * 2, base * 4), _cba(base * 4, base * 8))
        self.pool = nn.MaxPool2d(2)
        self.bott = _cba(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = _cba(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = _cba(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = _cba(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = _cba(base * 2, base)
        self.head_v = nn.Conv2d(base, 1, 1)
        self.head_f = nn.Conv2d(base, 1, 1)
        if attention:
            self.g4, self.g3 = _Gate(base * 8), _Gate(base * 4)
            self.g2, self.g1 = _Gate(base * 2), _Gate(base)

    def _skip(self, up, enc, gate):
        if self.attention:
            enc = gate(up, enc)
        return torch.cat([up, enc], 1)

    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2)); s4 = self.e4(self.pool(s3))
        b = self.bott(self.pool(s4))
        d = self.d4(self._skip(self.u4(b), s4, getattr(self, "g4", None)))
        d = self.d3(self._skip(self.u3(d), s3, getattr(self, "g3", None)))
        d = self.d2(self._skip(self.u2(d), s2, getattr(self, "g2", None)))
        d = self.d1(self._skip(self.u1(d), s1, getattr(self, "g1", None)))
        return {"vessel": self.head_v(d), "faz": self.head_f(d)}


class _Gate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.wg, self.wx, self.psi = nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, 1, 1)

    def forward(self, g, x):
        return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))


_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


@register("unet")
def _b_unet(cfg):
    return _UNet(cfg["model"].get("in_channels", 1), attention=False)


@register("attn_unet")
def _b_attn(cfg):
    return _UNet(cfg["model"].get("in_channels", 1), attention=True)


def build_baseline(cfg):
    name = cfg["model"]["name"]
    if name in _REGISTRY:
        return _REGISTRY[name](cfg)
    raise NotImplementedError(
        f"Baseline '{name}' uses the authors' official code; see configs/baseline_{name}.yaml "
        f"(`impl_source`) and register it in vgatnet/models/baselines.py.")
