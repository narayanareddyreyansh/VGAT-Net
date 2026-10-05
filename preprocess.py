import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


def clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """Contrast-Limited Adaptive Histogram Equalization (X_c = CLAHE(X))."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.ptp() + 1e-8)).astype(np.uint8)
    if _HAS_CV2:
        op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return op.apply(img)
    # numpy fallback: global histogram equalization
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
    return cdf[img]


def minmax_normalize(image):
    """X_n = (X_c - min) / (max - min)  ->  [0, 1]   (Eq. 2)."""
    img = np.asarray(image).astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def preprocess(image, clip_limit=2.0, tile_grid=(8, 8)):
    """Full pipeline used at train and test time."""
    return minmax_normalize(clahe(image, clip_limit, tile_grid))
