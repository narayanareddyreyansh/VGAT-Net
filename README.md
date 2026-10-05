# VGAT-Net

**A Vessel-Guided Anatomy-Aware Transformer Network for Joint Retinal Vessel and Foveal Avascular Zone Segmentation in OCT Angiography**

Implementation accompanying the paper (International Journal of Intelligent Engineering
and Systems). VGAT-Net jointly segments retinal vessels and the foveal avascular zone (FAZ)
from OCT-A images by using continuity-aware vessel representations as a directional
guidance signal for the FAZ, and derives clinical biomarkers from the predicted masks.

## Components

- **Multi-Scale Vessel Topology Encoder (MVTE)** — parallel 3×3 / 5×5 / dilated branches (Sec. 3.2)
- **Vessel Continuity Attention Module (VCAM)** — channel and spatial attention (Sec. 3.3)
- **Transformer Context Aggregator (TCA)** — long-range context at the bottleneck (Sec. 3.4)
- **Vessel-Guided FAZ Attention (VGFA)** — cross-attention from vessel to FAZ features (Sec. 3.5)
- **Cross-Task Interaction Block (CTIB)** — vessel/FAZ feature sharing (Sec. 3.6)
- **Topology-preserving hybrid loss** — Dice + BCE + soft-skeleton topology + boundary terms (Sec. 3.8)

## Repository layout

```
VGAT-Net/
├── configs/                 # YAML configs (VGAT-Net on OCTA-500 and ROSE-O, baselines)
├── data/
│   ├── splits/              # train / val / test id lists for OCTA-500 and ROSE-O
│   └── README.md            # dataset sources, folder layout, split protocol
├── results/
│   └── roseo/               # per-seed per-image test metrics, training logs and
│                            # the configuration used for the ROSE-O runs
├── scripts/
│   ├── prepare_roseo.py     # converts the ROSE-O release and writes its split files
│   ├── make_splits.py       # subject-level split generation for OCTA-500
│   ├── train.py             # training (one run per seed)
│   ├── evaluate.py          # mean +/- std over checkpoints on the test split
│   ├── per_image_metrics.py # per-image test metrics -> CSV (input of make_tables.py)
│   ├── make_tables.py       # result tables and paired significance tests from the CSVs
│   ├── predict.py           # inference on a folder of images
│   └── extract_biomarkers.py# vessel density, perfusion density, FAZ area / perimeter / circularity
├── vgatnet/
│   ├── models/              # vgatnet.py, modules.py, baselines.py
│   ├── data/                # dataset.py, preprocess.py (CLAHE + normalisation), transforms.py
│   ├── losses.py, metrics.py, biomarkers.py, utils.py
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/narayanareddyreyansh/VGAT-Net.git
cd VGAT-Net
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data

See [`data/README.md`](data/README.md) for the dataset sources, the folder layout and
the split protocol. OCTA-500 splits are regenerated with `scripts/make_splits.py`;
ROSE-O is converted and split with `scripts/prepare_roseo.py`.

## Training

One run per seed; the protocol of the paper (Adam, lr 1e-4, weight decay 1e-5,
cosine schedule, 100 epochs, early stopping with patience 15 on the validation loss,
mixed precision) is fixed in the YAML files.

```bash
# OCTA-500
python scripts/train.py --config configs/vgatnet_octa500.yaml --seed 0
python scripts/train.py --config configs/vgatnet_octa500.yaml --seed 1
python scripts/train.py --config configs/vgatnet_octa500.yaml --seed 2

# ROSE-O
python scripts/train.py --config configs/vgatnet_roseo.yaml --seed 0
```

Each run writes `runs/<experiment>/seed<k>/` with `best.pth` (lowest validation loss),
`last.pth`, `train_log.json` and `config_used.json`.

### Baselines

U-Net and Attention U-Net are implemented natively in `vgatnet/models/baselines.py`
and trained with the same script and protocol:

```bash
python scripts/train.py --config configs/baseline_unet.yaml --seed 0
python scripts/train.py --config configs/baseline_attn_unet.yaml --seed 0
```

For TransUNet, Swin-UNet, MISSFormer and the OCTA-specific methods, the official
implementation sources are listed in each `configs/baseline_*.yaml` (`impl_source`);
register a builder in `baselines.py` to plug them in. Every baseline model must return
a dict `{"vessel": logits, "faz": logits}`.

## Evaluation

```bash
# mean +/- std over checkpoints
python scripts/evaluate.py --config configs/vgatnet_roseo.yaml \
    --checkpoints runs/vgatnet_roseo/seed0/best.pth runs/vgatnet_roseo/seed1/best.pth runs/vgatnet_roseo/seed2/best.pth

# per-image test metrics of one checkpoint (DSC, IoU, SE, SP, HD95 for vessel and FAZ)
python scripts/per_image_metrics.py --config configs/vgatnet_roseo.yaml \
    --checkpoint runs/vgatnet_roseo/seed0/best.pth --out runs/vgatnet_roseo/seed0/per_image_test.csv

# tables and paired tests from the per-image CSVs
python scripts/make_tables.py summary runs/vgatnet_roseo
python scripts/make_tables.py compare runs/vgatnet_octa500 runs/baseline_unet_octa500 --task vessel
```

`make_tables.py compare` averages the per-image metrics over seeds, runs a two-tailed
paired t-test per metric (DSC, IoU, HD95), applies the Holm–Bonferroni correction
within that family and reports Cohen's d for paired samples, which is the procedure of
the statistical analysis in the paper.

## Released results

`results/roseo/` contains, for each of the three ROSE-O seeds, the per-image test
metrics (`per_image_test.csv`), the training log and the exact configuration used,
together with `test_results.json` (mean and standard deviation over the seeds).
The corresponding checkpoints (`best.pth`, seeds 0–2) are attached to the tagged
release as assets. OCTA-500 results are added in the same form when their runs are
complete.

## Inference and biomarkers

```bash
python scripts/predict.py --config configs/vgatnet_octa500.yaml \
    --checkpoint runs/vgatnet_octa500/seed0/best.pth --input path/to/images/ --output predictions/

python scripts/extract_biomarkers.py --vessel-dir predictions/vessel --faz-dir predictions/faz \
    --fov-mm 3.0 --size-px 512 --out biomarkers.csv --reference expert_biomarkers.csv
```

Vessel density is computed as the percentage of image pixels on the morphological
skeleton of the binarised vessel mask, perfusion density as the percentage of vessel
pixels, and the FAZ area, perimeter and circularity from the largest connected FAZ
component (see `vgatnet/biomarkers.py`).

## Citation

```bibtex
@article{vgatnet,
  title   = {A Vessel-Guided Anatomy-Aware Transformer Network for Joint Retinal Vessel and
             Foveal Avascular Zone Segmentation in OCT Angiography},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)).
