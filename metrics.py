import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _binarize(x, thr=0.5):
    return (np.asarray(x) >= thr).astype(np.uint8)


def dsc(pred, gt, eps=1e-6):
    p, g = _binarize(pred), _binarize(gt)
    inter = (p & g).sum()
    return (2 * inter + eps) / (p.sum() + g.sum() + eps)


def iou(pred, gt, eps=1e-6):
    p, g = _binarize(pred), _binarize(gt)
    inter = (p & g).sum()
    union = (p | g).sum()
    return (inter + eps) / (union + eps)


def sensitivity(pred, gt, eps=1e-6):
    p, g = _binarize(pred), _binarize(gt)
    tp = (p & g).sum()
    fn = ((1 - p) & g).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(pred, gt, eps=1e-6):
    p, g = _binarize(pred), _binarize(gt)
    tn = ((1 - p) & (1 - g)).sum()
    fp = (p & (1 - g)).sum()
    return (tn + eps) / (tn + fp + eps)


def hd95(pred, gt):
    """95th-percentile Hausdorff Distance in pixels (Eq. 25)."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy required for HD95")
    p, g = _binarize(pred), _binarize(gt)
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(1 - g)
    dt_p = distance_transform_edt(1 - p)
    # boundary via morphological gradient
    from scipy.ndimage import binary_erosion
    bp = p & (~binary_erosion(p))
    bg = g & (~binary_erosion(g))
    d_pg = dt_g[bp.astype(bool)]
    d_gp = dt_p[bg.astype(bool)]
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


ALL_METRICS = ["DSC", "IoU", "SE", "SP", "HD95"]


def compute_all(pred, gt):
    return {
        "DSC": dsc(pred, gt) * 100,
        "IoU": iou(pred, gt) * 100,
        "SE": sensitivity(pred, gt) * 100,
        "SP": specificity(pred, gt) * 100,
        "HD95": hd95(pred, gt),
    }


class MetricAccumulator:
    def __init__(self):
        self.store = {m: [] for m in ALL_METRICS}

    def update(self, pred, gt):
        r = compute_all(pred, gt)
        for k, v in r.items():
            if not (isinstance(v, float) and np.isnan(v)):
                self.store[k].append(v)

    def summary(self):
        return {k: (float(np.mean(v)) if v else float("nan"),
                    float(np.std(v)) if v else float("nan"))
                for k, v in self.store.items()}
