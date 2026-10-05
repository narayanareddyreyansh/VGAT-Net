import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prob, target, eps=1e-6):
    """Eq. 18."""
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def bce_loss(logits, target):
    """Eq. 19."""
    return F.binary_cross_entropy_with_logits(logits, target)


def soft_skeleton(x, iters=10):
    """Differentiable soft-skeleton (soft-clDice) used by the topology operator Psi."""
    def soft_erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(img):
        return F.max_pool2d(img, (3, 3), 1, (1, 1))

    def soft_open(img):
        return soft_dilate(soft_erode(img))

    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def topology_loss(prob, target, iters=10):
    """Eq. 20 (vessel branch). L2 between soft skeletons Psi(Y_v), Psi(G_v)."""
    return F.mse_loss(soft_skeleton(prob, iters), soft_skeleton(target, iters))


def _boundary_map(x):
    """Sobel-based boundary extraction."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return torch.clamp(g, 0, 1)


def boundary_loss(prob, target, eps=1e-6):
    """Eq. 20 (FAZ branch). 1 - soft-IoU of boundary maps."""
    bp = _boundary_map(prob)
    bg = _boundary_map(target)
    inter = (bp * bg).sum(dim=(1, 2, 3))
    union = (bp + bg - bp * bg).sum(dim=(1, 2, 3)) + eps
    return (1 - inter / union).mean()


class TopologyPreservingHybridLoss(nn.Module):
    """
    L_v = L_dice + L_bce + L_topo          (Eq. 21)
    L_f = L_dice + L_boundary              (Eq. 21)
    L   = lambda1 * L_v + lambda2 * L_f    (Eq. 22)
    """

    def __init__(self, lambda1=1.0, lambda2=0.8, topo_iters=10):
        super().__init__()
        self.l1 = lambda1
        self.l2 = lambda2
        self.topo_iters = topo_iters

    def forward(self, outputs, targets):
        v_logits, f_logits = outputs["vessel"], outputs["faz"]
        v_gt, f_gt = targets["vessel"], targets["faz"]
        v_prob = torch.sigmoid(v_logits)
        f_prob = torch.sigmoid(f_logits)

        l_v = (dice_loss(v_prob, v_gt)
               + bce_loss(v_logits, v_gt)
               + topology_loss(v_prob, v_gt, self.topo_iters))
        l_f = dice_loss(f_prob, f_gt) + boundary_loss(f_prob, f_gt)

        total = self.l1 * l_v + self.l2 * l_f
        return total, {"loss": total.item(), "loss_vessel": l_v.item(),
                       "loss_faz": l_f.item()}
