import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class Augmentor:
    def __init__(self, rotation=15, scale=(0.9, 1.1), gamma=(0.8, 1.2),
                 elastic_alpha=20, elastic_sigma=4, p=0.5, seed=0):
        self.rotation = rotation
        self.scale = scale
        self.gamma = gamma
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p = p
        self.rng = np.random.RandomState(seed)

    def _maybe(self):
        return self.rng.rand() < self.p

    def __call__(self, image, masks):
        if self._maybe():
            image = np.fliplr(image).copy()
            masks = [np.fliplr(m).copy() for m in masks]
        if self._maybe():
            image = np.flipud(image).copy()
            masks = [np.flipud(m).copy() for m in masks]
        if _HAS_CV2 and self._maybe():
            image, masks = self._affine(image, masks)
        if _HAS_CV2 and self._maybe():
            image, masks = self._elastic(image, masks)
        if self._maybe():
            g = self.rng.uniform(*self.gamma)
            image = np.clip(image, 0, 1) ** g
        return image, masks

    def _affine(self, image, masks):
        h, w = image.shape[:2]
        angle = self.rng.uniform(-self.rotation, self.rotation)
        s = self.rng.uniform(*self.scale)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        masks = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT) for m in masks]
        return image, masks

    def _elastic(self, image, masks):
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur((self.rng.rand(h, w) * 2 - 1).astype(np.float32),
                              (0, 0), self.elastic_sigma) * self.elastic_alpha
        dy = cv2.GaussianBlur((self.rng.rand(h, w) * 2 - 1).astype(np.float32),
                              (0, 0), self.elastic_sigma) * self.elastic_alpha
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        mapx = (xx + dx).astype(np.float32)
        mapy = (yy + dy).astype(np.float32)
        image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        masks = [cv2.remap(m, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT) for m in masks]
        return image, masks
