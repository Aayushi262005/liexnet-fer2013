# Which cheap additions help a LiExNet-style network on FER2013?

Code, per-seed results and statistics for the paper
**"Lightweight Facial Expression Recognition on FER2013: Which Cheap Additions Help a LiExNet-Style Network?"**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/liexnet-fer2013/blob/main/notebooks/LiExNet_Improved_v3.ipynb)

We train a compact LiExNet-style CNN (48x48 grayscale input, four blocks, ECA channel attention) under one controlled
pipeline and ask which inexpensive additions improve FER2013 accuracy once seed variance is accounted for:

- a depthwise **spatial gate** and an **SK-lite** block (variant D, on top of the ECA baseline A);
- **knowledge distillation** (KD) from a ResNet-18 teacher (variant E = D + KD).

Each variant is trained at two widths with 3-5 paired seeds. Everything is in a single notebook.

## Main results

Held-out **test half** (3,589 images), mean +/- sample std over seeds. "TTA" = horizontal-flip test-time augmentation.

| Variant | Params | Seeds | Test acc (%) | Test acc, TTA (%) |
|---|---:|:-:|---:|---:|
| A (ECA baseline) | 289,903 | 5 | 67.59 +/- 0.17 | 68.66 +/- 0.22 |
| D (+ gate, SK-lite) | 299,968 | 5 | 67.26 +/- 0.75 | 68.48 +/- 0.47 |
| E (D + KD) | 299,968 | 5 | 68.02 +/- 0.09 | 69.30 +/- 0.27 |
| A-wide | 501,049 | 3 | 68.27 +/- 0.24 | 69.09 +/- 0.68 |
| D-wide | 515,333 | 3 | 67.99 +/- 0.45 | 69.34 +/- 0.49 |
| E-wide (D + KD) | 515,333 | 3 | 69.23 +/- 0.20 | 70.50 +/- 0.65 |
| Teacher (ResNet-18, 11.18 M) | 11.18 M | 1 | 70.52 | 71.16 |

Paired differences (same seed, percentage points, 95% t-interval):

| Effect | n | Mean diff | 95% CI |
|---|:-:|---:|---|
| Architecture, narrow (D - A) | 5 | -0.33 | [-1.28, +0.61] |
| KD, narrow (E - D) | 5 | +0.77 | [-0.15, +1.69] |
| KD, wide (E - D) | 3 | +1.24 | [+0.14, +2.33] |

Take-aways: the spatial gate + SK-lite gave no detectable gain over ECA alone; KD has a positive estimate, with an interval
that excludes zero only for the wide student (n = 3, uncorrected for multiple comparisons). With 3,589 test images the
binomial standard error of accuracy is about 0.8 pp, so differences below ~1 pp should not be over-interpreted.
See the paper for the full tables, per-class F1 and confusion matrices.

## Important notes on the protocol

- **This is not the official PrivateTest evaluation.** The CSV used here contains only `Training` (28,709) and
  `PrivateTest` (7,178) rows. The 7,178 held-out images are split 50/50, stratified by class with seed 42, into a
  validation half (checkpoint selection) and a test half (reported numbers, used once per run).
- Numbers are **not comparable** to LiExNet's 79.2% (processed FER2013, 128x128 RGB, 5-fold CV, no augmentation).
- Latency and FLOPs are Colab proxies, not embedded-device measurements. FLOPs are counted by PyTorch's
  `FlopCounterMode` (= 2 x multiply-accumulates, conv/linear only).

## Repository layout

```
notebooks/LiExNet_Improved_v3.ipynb   # the full experiment (data -> training -> statistics -> efficiency)
results/                              # per-seed results, tables, confusion matrices (see results/README.md)
requirements.txt
LICENSE
```

## Reproducing the results

1. **Get FER2013 as a CSV** with columns `emotion`, `pixels`, `Usage` (the version used here has only `Training` and
   `PrivateTest` rows). The dataset is not redistributed here; please obtain it from its original source and respect its terms.
2. **Open the notebook in Google Colab** (badge above) and select a **GPU runtime**.
3. Put the CSV in `/content/` as `FER.csv`, or anywhere in your Google Drive under the name `FER.csv`
   (the notebook searches Drive automatically).
4. **Run all cells.** Outputs are written to `MyDrive/liexnet_v3/`:
   `runs.csv`, `probs/`, `teacher_resnet18.pt`, `master_table.csv`, `per_class_f1.csv`, paired-statistics CSVs,
   `confusion_matrices.png`, `config.json`.
5. The notebook is **resumable**: every finished run is appended to `runs.csv`, the teacher is cached, and a re-run skips
   completed runs. If Colab disconnects, reconnect and run all cells again.

Total: 6 variants, 24 student runs (narrow: seeds 0-4 for A, D, E; wide: seeds 0-2 for A, D, E), plus one teacher.

To regenerate the paper's tables from the stored results without retraining, keep `runs.csv` and `probs/` in the output
folder and run all cells: finished runs are skipped. Note that cell 10 re-measures latency, so the milliseconds will differ
slightly from the paper; the paper's Table 6 values are in `results/master_table.csv`.

## Implementation details

| Item | Setting |
|---|---|
| Input | 48x48 grayscale, normalised with training-set mean/std |
| Student blocks | conv3x3 -> depthwise 3x3 -> attention -> conv3x3 -> 2x2 max-pool; widths 24/48/72/96 (narrow) or 32/64/96/128 (wide) |
| Head | global average pool -> FC-256 + ReLU -> dropout 0.3 -> FC-7 |
| ECA kernel | odd(abs(log2(C)/2 + 1)), i.e. 3 or 5 |
| Variant D | ECA gate 2*sigmoid with zero-initialised weights (identity at start), depthwise 5x5 spatial gate (also identity-initialised), SK-lite replacing the depthwise layer in block 3 |
| Optimiser | AdamW, lr 2e-3, weight decay 5e-4 (all parameters), batch 128, 60 epochs, 3 warm-up epochs then cosine to 1% |
| Regularisation | label smoothing 0.1, sqrt inverse-frequency class weights, gradient clipping 5, mixed precision (CUDA) |
| EMA | decay 0.999 with warm-up `min(0.999, (1+t)/(10+t))`; best-epoch EMA weights chosen on the validation half |
| Augmentation (GPU) | rotation +/-10 deg, scale +/-10%, translation +/-5% of image width, horizontal flip, contrast +/-20%, brightness +/-0.1, random erasing (p = 0.25, 2-15% area) |
| KD | T = 4, alpha = 0.5, `L = (1-a)*CE + a*T^2*KL(teacher || student)` (KD term not class-weighted) |
| Teacher | ResNet-18 (ImageNet weights when available), 96x96 bilinear-upsampled input, 30 epochs, AdamW lr 5e-4 / wd 1e-4, trained once and cached |
| Seeds | narrow 0-4, wide 0-2; the model is seeded before construction, so D and E share identical initial weights |
| Statistics | paired t-interval and paired t-test over seeds; no multiple-comparison correction |

## Limitations

- One fixed 50/50 split of the held-out pool, FER2013 only.
- 3-5 seeds give wide intervals; p-values are a guide, not proof.
- The teacher is a single run (no seed variance).
- Confidence intervals reflect seed variance only, not test-set sampling noise.
- Latency/FLOPs are Colab proxies.

## Citation

```bibtex
@misc{liexnet_fer2013,
  title  = {Lightweight Facial Expression Recognition on FER2013: Which Cheap Additions Help a LiExNet-Style Network?},
  author = {YOUR NAME},
  year   = {2026},
  note   = {Code: https://github.com/YOUR_USERNAME/liexnet-fer2013}
}
```

## References

- LiExNet: Ramirez-Quintana et al., *Sensors* 25(23), 7264, 2025.
- ECA-Net: Wang et al., CVPR 2020. SK-Net: Li et al., CVPR 2019. Knowledge distillation: Hinton et al., 2015.
- FER2013: Goodfellow et al., *Neural Networks* 64, 2015.

## License

Code released under the MIT License (see `LICENSE`). FER2013 is subject to its own terms and is not included.
