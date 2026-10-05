"""VGAT-Net: Vessel-Guided Anatomy-Aware Transformer Network (Sec. 3, Eq. 1)."""
import torch.nn as nn

from .modules import (
    MultiScaleVesselTopologyEncoder,
    VesselContinuityAttentionModule,
    TransformerContextAggregator,
    VesselGuidedFAZAttention,
    CrossTaskInteractionBlock,
    SegmentationDecoder,
)


class VGATNet(nn.Module):
    """
    (Y_v, Y_f) = D(C(G(T(V(E(X))))))        [Eq. 1]

    Config keys (configs/*.yaml -> model):
        in_channels, base_channels, encoder_depth,
        embed_dim, tca_depth, tca_heads, patch_size, vgfa_heads.
    """

    def __init__(self, in_channels=1, base_channels=64, encoder_depth=4,
                 embed_dim=512, tca_depth=6, tca_heads=8, patch_size=2,
                 vgfa_heads=8):
        super().__init__()
        c = base_channels
        self.encoder = MultiScaleVesselTopologyEncoder(in_channels, c, encoder_depth)   # E
        self.vcam = VesselContinuityAttentionModule(c, r=16, kernel_size=7)             # V
        self.tca = TransformerContextAggregator(c, embed_dim, tca_depth, tca_heads,
                                                mlp_ratio=4.0, patch_size=patch_size)   # T
        self.vgfa = VesselGuidedFAZAttention(c, heads=vgfa_heads)                       # G
        self.ctib = CrossTaskInteractionBlock(c)                                        # C
        self.vessel_decoder = SegmentationDecoder(c, num_stages=encoder_depth)          # D_v
        self.faz_decoder = SegmentationDecoder(c, num_stages=encoder_depth)             # D_f

    def forward(self, x):
        out_size = x.shape[-2:]
        bottleneck, skips = self.encoder(x)      # E(X)
        f_v = self.vcam(bottleneck)              # V(.)
        f_t = self.tca(f_v)                      # T(.)
        vessel_feat, faz_feat = self.vgfa(f_t)   # G(.) -> (F_v, F_faz)
        fused = self.ctib(vessel_feat, faz_feat)  # C(.) shared representation
        vessel_logits = self.vessel_decoder(fused, skips, out_size)   # D_v
        faz_logits = self.faz_decoder(fused, skips, out_size)         # D_f
        return {"vessel": vessel_logits, "faz": faz_logits}


def build_model(cfg):
    m = cfg["model"]
    return VGATNet(
        in_channels=m.get("in_channels", 1),
        base_channels=m.get("base_channels", 64),
        encoder_depth=m.get("encoder_depth", 4),
        embed_dim=m.get("embed_dim", 512),
        tca_depth=m.get("tca_depth", 6),
        tca_heads=m.get("tca_heads", 8),
        patch_size=m.get("patch_size", 2),
        vgfa_heads=m.get("vgfa_heads", 8),
    )
