import numpy as np

try:
    from skimage.morphology import skeletonize
    from skimage.measure import label, regionprops, find_contours
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


def mm_per_pixel(fov_mm=3.0, size_px=512):
    return fov_mm / size_px


def vessel_density(vessel_mask, thr=0.5):
    """VD: skeleton length fraction (%)."""
    m = (np.asarray(vessel_mask) >= thr).astype(np.uint8)
    skel = skeletonize(m > 0)
    return 100.0 * skel.sum() / m.size


def perfusion_density(vessel_mask, thr=0.5):
    """PD: vessel-area fraction (%)."""
    m = (np.asarray(vessel_mask) >= thr).astype(np.uint8)
    return 100.0 * m.sum() / m.size


def _largest_component(mask):
    lbl = label(mask)
    if lbl.max() == 0:
        return mask
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return (lbl == counts.argmax()).astype(np.uint8)


def faz_metrics(faz_mask, mm_px, thr=0.5):
    """Return FAZ area (mm^2), perimeter (mm), circularity index."""
    m = (np.asarray(faz_mask) >= thr).astype(np.uint8)
    m = _largest_component(m)
    if m.sum() == 0:
        return {"FAZ_area_mm2": 0.0, "FAZ_perimeter_mm": 0.0, "FAZ_circularity": 0.0}
    props = regionprops(m)[0]
    area = props.area * (mm_px ** 2)                  # A = N_px * s^2
    # perimeter from contour
    contours = find_contours(m, 0.5)
    if contours:
        c = max(contours, key=len)
        d = np.diff(c, axis=0)
        perim = np.sqrt((d ** 2).sum(axis=1)).sum() * mm_px
    else:
        perim = props.perimeter * mm_px
    ci = (4 * np.pi * area) / (perim ** 2 + 1e-8)     # CI = 4*pi*A / P^2
    return {"FAZ_area_mm2": float(area),
            "FAZ_perimeter_mm": float(perim),
            "FAZ_circularity": float(ci)}


def extract_all(vessel_mask, faz_mask, fov_mm=3.0, size_px=512, thr=0.5):
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for biomarker extraction")
    mm_px = mm_per_pixel(fov_mm, size_px)
    out = {
        "VD_percent": float(vessel_density(vessel_mask, thr)),
        "PD_percent": float(perfusion_density(vessel_mask, thr)),
    }
    out.update(faz_metrics(faz_mask, mm_px, thr))
    return out
