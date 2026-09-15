# Pneumonia Detection from Chest X-Rays: Does a More Accurate Model Mean a More Trustworthy One?

A study of whether CNN confidence on chest X-ray classification is calibrated — i.e. whether a
predicted probability of "90% pneumonia" actually corresponds to being right 90% of the time —
and whether a stronger, pretrained model is automatically a more trustworthy one just because
it's more accurate.

## Motivation

A clinician using a model's confidence score to triage cases needs that confidence to mean what
it says. This project directly extends the research direction of Prof. Narayanan C. Krishnan
("CK") at IIT Palakkad, whose published work *"Understanding Calibration of Deep Neural Networks
for Medical Image Classification"* (Sambyala, Niyaza, Krishnan, Bathula, 2023) studies exactly
this question in the medical imaging setting.

This is the second half of a two-part portfolio; the first, on credit card fraud detection, asks
the same question for classical convex-optimization classifiers (Logistic Regression, Naive
Bayes, an SVM implemented from scratch via SMO) under severe class imbalance.

## Research question

1. Is a CNN trained to classify pneumonia from chest X-rays well-calibrated out of the box?
2. Does post-hoc temperature scaling meaningfully close any calibration gap?
3. Does a more accurate model (a fine-tuned, pretrained ResNet18) also produce more trustworthy
   confidence scores than a smaller model trained from scratch — or can these two properties
   diverge?

## Reproducibility note — read this before citing any single number below

Repeated runs of this notebook (identical code, `seed=42`) produced **noticeably different**
calibration numbers, particularly for the from-scratch custom CNN:

| Metric | Run A | Run B (canonical — used below) |
|---|---|---|
| Custom CNN test accuracy | 83.33% | 81.25% |
| Custom CNN ECE (uncalibrated) | 5.98% | 17.74% |
| Custom CNN MCE (uncalibrated) | 17.97% | 62.16% |
| Custom CNN Normal-class recall | 85.5% | 50.9% |
| ResNet18 test accuracy | 85.58% | 83.81% |
| ResNet18 ECE (uncalibrated) | 15.02% | 16.08% |

The train/val/test *split* is fixed (`random_state=42`, identical counts both runs). The
variance comes from training dynamics — GPU operation nondeterminism beyond what
`cudnn.deterministic` covers, and unseeded `DataLoader` worker shuffling — compounding over
training into different early-stopping checkpoints, especially for the from-scratch model, whose
per-epoch validation loss is visibly noisier than ResNet18's (see training logs in the
notebook). This is reported explicitly rather than silently picking whichever run looked best.
**One finding from this variance itself is worth stating plainly: the from-scratch CNN's
calibration is considerably less stable across runs than the fine-tuned ResNet18's — training
stability is not just an accuracy question, it's a calibration-reliability question too.**

All results below are from Run B, the notebook's current committed state.

## Key findings

**1. ResNet18 consistently trades pneumonia recall for a severe, stable false-positive rate on
healthy patients — this direction holds across both runs.** ResNet18 achieves near-perfect
pneumonia recall (99.7% both runs) but flags a large, consistent fraction of *healthy* patients
as diseased: 38.0% in Run A, 42.7% in Run B. This is the one finding in this project that did
not move between runs, and it's the clinically important one: a model that essentially never
misses real disease but is wrong about more than a third of healthy patients has a real,
predictable failure mode, not a calibration curiosity.

**2. Which model is "better calibrated" is not a stable claim — and that instability is itself
informative.** By Maximum Calibration Error, the custom CNN was better than ResNet18 in both
runs (17.97% vs. 64.31% in Run A; 52.81% vs. 75.71% in Run B, after temperature scaling) — this
direction held. By Expected Calibration Error, it did not: the custom CNN was far better in Run A
(5.98% vs. 15.02%) but slightly *worse* in Run B (17.44% vs. 16.07%). Reporting only Run A, as an
earlier draft of this project did, would have overstated a clean "hand-built model is better
calibrated" story that a second run doesn't support. The honest version: ResNet18's calibration
is the more *stable* of the two, even though the custom CNN's is sometimes better in absolute
terms — a real, if less tidy, finding about what fine-tuning from a strong pretrained backbone
buys you beyond raw accuracy.

**3. Temperature scaling does not fix minority-class calibration in either model, in either
run.** Minority-class (Normal) ECE stayed high after fitting temperature scaling in every
observed case (Run B: 50.35% → 50.38% for the custom CNN, 43.99% → 44.00% for ResNet18 — both
essentially unchanged). This is a known, real limitation of global temperature scaling, not an
implementation bug: one shared scalar can only rescale overall confidence sharpness, not correct
a bias that differs systematically by class. The direct next step — and the analog of the
asymmetric class-weighting already used for the SVM in the companion fraud-detection project —
is per-class (vector) temperature scaling, not attempted here.

**4. The from-scratch NumPy convolution and max-pooling implementation is verified correct**
against PyTorch autograd to floating-point tolerance — forward pass, input gradient, weight
gradient, and bias gradient all match exactly.

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kaggle (paultimothymooney).
- **Two dataset quirks caught and corrected, rather than silently inherited:**
  - The provided `val/` split has only 16 images — too small to reliably fit a temperature
    scaling parameter or make early-stopping decisions. `train/` and `val/` were merged and
    **re-split 85/15 with stratification** (4,447 train / 785 val, class ratios preserved). The
    official `test/` split (624 images) was left completely untouched.
  - The official test set has a **different class ratio than train/val** — 390 Pneumonia :
    234 Normal (≈1.67:1) vs. ≈2.9:1 in train — a real distribution shift between splits.
- Class imbalance handled via `pos_weight` in `BCEWithLogitsLoss` (≈0.35, computed from the
  training split), the direct analog of the asymmetric class-weighting used for the SVM in the
  fraud-detection project.

## Methodology

### Models
- **Custom CNN, trained from scratch:** 4 conv blocks (`Conv2d → BatchNorm2d → ReLU →
  MaxPool2d`), channel widths 16→32→64→128, global average pooling, single-logit linear head.
- **ResNet18, fine-tuned from ImageNet weights**, final FC layer replaced with a single-logit
  linear head.
- Both trained with `BCEWithLogitsLoss` (class-weighted), Adam (`lr=1e-3` custom CNN, `1e-4`
  ResNet18), early stopping on validation loss (patience 5).

### From-scratch verification
`NumPyConv2D` / `NumPyMaxPool2D` implement forward and backward passes for convolution and max
pooling in pure NumPy. Verified against an identically-initialized PyTorch model:

| Check | Result |
|---|---|
| Forward pass | Match |
| Input gradient (dX) | Match |
| Weight gradient (dW) | Match |
| Bias gradient (db) | Match |

### Calibration
- **Expected Calibration Error (ECE)** and **Maximum Calibration Error (MCE)**, computed
  directly (10-bin), not via a library.
- **Temperature scaling:** single scalar `T`, fit via LBFGS minimizing NLL on the held-out
  validation set (never test), applied as `logits / T` before the sigmoid.
- **Minority-class calibration** computed separately on the Normal (minority) test subset.
- **Grad-CAM** on both models' final conv layers.

## Results (Run B — canonical, see Reproducibility Note above)

| Model | Accuracy | ROC-AUC | Recall (Pneumonia) | Precision (Pneumonia) | Optimal T | ECE Before | ECE After | MCE Before | MCE After | Minority ECE Before | Minority ECE After |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | 81.25% | 0.9381 | 99.49% | 77.14% | 1.186 | 17.74% | 17.44% | 62.16% | 52.81% | 50.35% | 50.38% |
| ResNet18 (pretrained) | 83.81% | 0.9538 | 99.74% | 79.55% | 1.018 | 16.08% | 16.07% | 76.08% | 75.71% | 43.99% | 44.00% |

Full per-class precision/recall/F1, confusion matrices, reliability diagrams, and Grad-CAM
overlays for both models are in `results/`.

**Grad-CAM, qualitative note:** on the two Normal-class examples shown, ResNet18's attention
concentrates heavily on central/mediastinal (heart and spine) regions rather than the lung
fields themselves, while the custom CNN's attention stays more on the lateral lung/rib areas —
worth a closer look (not a strong claim here) as a possible partial explanation for ResNet18's
high false-positive rate on healthy patients: attending to anatomy other than the lungs on
"Normal" cases is a plausible route to a spurious pneumonia call.

## Discussion

The two models represent a real trade-off, not a strict ranking, and the instability across runs
is itself part of the finding rather than noise to explain away. ResNet18's near-perfect
pneumonia recall is attractive if "never miss a case" is the only goal — but it consistently
achieves this by being wrong about a large fraction of healthy patients, and its confidence
scores are not a reliable guide to when it's one of those wrong cases (MCE 75.71%: in its
worst-calibrated confidence bucket, stated confidence and actual accuracy differ by 76
percentage points). The custom CNN's calibration is sometimes better in absolute terms but is
demonstrably less stable across training runs — a real cost that a single reported run would
hide.

Temperature scaling, applied naively, does not resolve either issue: it is a single global
correction, and the miscalibration here is concentrated in the minority class in both models,
in both runs. This mirrors the class-imbalance-and-calibration throughline of the companion
fraud-detection project. The consistent lesson across both projects: handling class imbalance in
the loss function does not automatically produce trustworthy, class-balanced *confidence* — that
has to be measured and corrected directly, per class, not assumed as a byproduct of accuracy.

## Repository structure

```
pneumonia-calibration/
├── README.md                                    # this file
├── data/                                         # .gitignored — not committed (Kaggle TOS + size)
├── notebooks/
│   └── pneumonia_calibration.ipynb               # full pipeline: data, models, training,
│                                                  # NumPy verification, calibration, Grad-CAM
├── results/
│   ├── gradcam_custom_cnn.png
│   ├── gradcam_resnet18.png
│   ├── reliability_diagrams_custom_cnn.png
│   ├── reliability_diagrams_resnet18.png
│   └── final_comparison_table.csv
└── requirements.txt
```

## How to run

**On Kaggle (recommended, matches how this was built):**
1. Click **"+ Add Data"** in the notebook editor's right sidebar → search
   `chest-xray-pneumonia` → Add. Mounts at `/kaggle/input/chest-xray-pneumonia/` automatically.
2. Enable a GPU (Settings → Accelerator).
3. Run all cells top to bottom.

**On Colab:**
1. Upload a `kaggle.json` API token (from kaggle.com/settings):
   ```
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   !unzip -q chest-xray-pneumonia.zip -d ./chest_xray_temp
   ```
2. Enable a GPU runtime.
3. Run all cells top to bottom.

## Limitations and future work

- **Run-to-run variance** (see Reproducibility Note) means any single run's calibration numbers
  should be read as one sample, not a definitive value — averaging over 3-5 seeds is the natural
  next step if this is extended further.
- **Per-class (vector) temperature scaling** to directly target the minority-class miscalibration
  that global scaling leaves unresolved in every observed run.
- **Dataset size and source**: a single public dataset from one institution; calibration behavior
  on X-rays from different hospitals/scanners is untested.

## Relation to the companion project

Together with the credit card fraud detection project (classical convex-optimization
classifiers, SVM via SMO derived and implemented from scratch, calibration under class
imbalance), this project asks the same question — is model confidence trustworthy? — for a
hand-specified representation plus convex optimization versus a learned CNN representation, both
under real-world class imbalance. The consistent finding across both: neither approach produces
trustworthy calibration as a free side effect of optimizing accuracy, and neither class
reweighting nor global temperature scaling reliably fixes it on its own.
