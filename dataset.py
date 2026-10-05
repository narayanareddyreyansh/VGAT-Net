
import os
import numpy as np

try:
    import cv2
    def _imread(p):
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    def _resize(a, size, nn=False):
        return cv2.resize(a, size, interpolation=cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR)
except Exception:
    from PIL import Image
    def _imread(p):
        return np.array(Image.open(p).convert("L"))
    def _resize(a, size, nn=False):
        im = Image.fromarray(a)
        return np.array(im.resize(size, Image.NEAREST if nn else Image.BILINEAR))

import torch
from torch.utils.data import Dataset

from .preprocess import preprocess
from .transforms import Augmentor


def read_split(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


class OCTASegDataset(Dataset):
    def __init__(self, root, split_file, image_size=512, train=False,
                 clip_limit=2.0, seed=0):
        self.root = root
        self.ids = read_split(split_file)
        self.size = image_size
        self.train = train
        self.clip_limit = clip_limit
        self.aug = Augmentor(seed=seed) if train else None

    def __len__(self):
        return len(self.ids)

    def _load(self, sub, _id):
        for ext in (".png", ".bmp", ".tif", ".jpg"):
            p = os.path.join(self.root, sub, _id + ext)
            if os.path.exists(p):
                return _imread(p)
        raise FileNotFoundError(f"missing {sub}/{_id}")

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img = self._load("images", _id)
        ves = self._load("vessel", _id)
        faz = self._load("faz", _id)

        size = (self.size, self.size)
        img = _resize(img, size, nn=False)
        ves = _resize(ves, size, nn=True)
        faz = _resize(faz, size, nn=True)

        img = preprocess(img, clip_limit=self.clip_limit)          # CLAHE + norm
        ves = (ves > 127).astype(np.float32)
        faz = (faz > 127).astype(np.float32)

        if self.train:
            img, (ves, faz) = self.aug(img, [ves, faz])

        img = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
        ves = torch.from_numpy(np.ascontiguousarray(ves)).float().unsqueeze(0)
        faz = torch.from_numpy(np.ascontiguousarray(faz)).float().unsqueeze(0)
        return {"image": img, "vessel": ves, "faz": faz, "id": _id}
