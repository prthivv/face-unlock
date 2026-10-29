# Face Unlock

A real-time facial recognition and biometric authentication system built with a **ResNet50** backbone (initialised from ImageNet weights) and **ArcFace loss**, trained on the **CASIA-WebFace** dataset. Built as a university machine learning project in two weeks.


## Overview

This project implements a complete face verification pipeline similar to smartphone Face ID:

- **Train** a deep embedding model to produce discriminative 512-D face representations
- **Enrol** a new user by capturing 10 webcam photos and storing their mean embedding
- **Verify** in real time: a live camera feed is compared against enrolled templates using cosine similarity

The architecture and training setup follows published research: ResNet backbone + ArcFace loss + CASIA-WebFace is a standard combination in face recognition papers.


## Architecture

```
Webcam Frame (BGR)
     │
     ├──► CLAHE contrast enhancement (detection copy only)
     │         │
     │         ▼
     │    MTCNN Face Detection → bounding box of largest face
     │         │
     ▼         ▼
Crop ORIGINAL frame: box + 14 px margin → 112×112 (PIL bilinear)
     │
     ▼
Normalise to [-1, 1]
     │
     ▼
ResNet50 Backbone (ImageNet pretrained, head replaced)
  └─ Linear(2048 → 512)
  └─ BatchNorm1d(512)
     │
     ▼
512-D Embedding (L2 normalised)
     │
     ▼
Cosine Similarity vs. Enrolled Templates
     │
     ├─ similarity > 0.60  →   UNLOCKED
     └─ similarity ≤ 0.60  →   UNKNOWN
```

**Loss function — ArcFace:**

```
L = −log [ e^(s·cos(θ + m)) / (e^(s·cos(θ+m)) + Σ e^(s·cos(θⱼ))) ]
```

where `s = 64` (scale) and `m = 0.1` (angular margin in radians). The margin forces same-identity embeddings to cluster more tightly on the unit hypersphere.

---

## Dataset

**Training:** [CASIA-WebFace](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface)
- ~494,000 images across 10,572 identities
- All images preprocessed with MTCNN → 112×112 face crops (detected box + 14 px margin)
- 489,409 images retained after face detection filtering

**Evaluation:** [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- `evaluate.py` currently uses the LFW DevTest pair lists (`matchpairsDevTest.csv` / `mismatchpairsDevTest.csv`)
- Pairs where either image failed face detection during preprocessing are skipped

---

## Project Structure

```
face-unlock/
├── data/
│   ├── raw/                  # Downloaded datasets (CASIA-WebFace, LFW)
│   ├── processed/            # MTCNN 112×112 face crops
│   └── register/             # Enrolled face templates (.npy)
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset class (reads casia-webface.txt)
│   │   └── preprocess.py     # MTCNN detection + cropping for the datasets
│   ├── models/
│   │   ├── backbone.py       # ResNet50 feature extractor
│   │   └── arcface.py        # ArcFace loss module
│   ├── face_utils.py         # Shared live-camera face detection + cropping
│   ├── train.py              # Training loop (AMP, StepLR, checkpointing)
│   ├── enroll.py             # Webcam enrolment script
│   ├── verify.py             # Real-time verification script
│   └── evaluate.py           # LFW benchmark evaluation
├── checkpoints/              # Saved model weights (Git LFS)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/prthivv/face-unlock.git
cd face-unlock
git lfs pull                     # downloads the checkpoint files
conda create -n faceunlock python=3.11
conda activate faceunlock
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify the environment:

```bash
pip check
python -c "import numpy, torch, cv2; print(numpy.__version__, torch.__version__, cv2.__version__)"
```

> **Note:** versions are constrained by `facenet-pytorch 2.6.0`, which requires `torch 2.2.x`, `torchvision 0.17.x`, `numpy < 2` and `Pillow < 10.3`. Don't upgrade NumPy to 2.x: it breaks `torch.from_numpy` with torch 2.2. Install only `opencv-python`, not `opencv-python-headless`, as the headless build disables `cv2.imshow`.

### 2. Download and preprocess the datasets

Download [CASIA-WebFace from Kaggle](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) and extract to `data/raw/casia-webface/`, and LFW to `data/raw/lfw/`.

Run MTCNN preprocessing:

```bash
python src/data/preprocess.py
```

> **All commands are run from the repository root** (not from `src/`), so that `checkpoints/` and `data/` resolve correctly.

---

## Training

```bash
python src/train.py
```

Checkpoints are saved to `checkpoints/` every 5 epochs, plus `best.pth` for the lowest average training loss. Training ran for 25 epochs on an RTX 3050 6 GB (~5 hours), reaching a final ArcFace loss of ~1.65.

**Key training settings:**

| Hyperparameter | Value |
|---|---|
| Optimiser | SGD (momentum=0.9, weight_decay=5e-4) |
| Learning Rate | 1e-3, StepLR ×0.1 at epoch 10 & 20 |
| Batch Size | 256 |
| Mixed Precision | `torch.cuda.amp` (GradScaler) |
| Embedding Dim | 512 |
| ArcFace scale `s` | 64 |
| ArcFace margin `m` | 0.1 rad |

---

## Usage

### Enrol a new face

```bash
python src/enroll.py
```

A live preview shows a green box and MTCNN's detection confidence when a face is found, or "No face detected" otherwise. Press `SPACE` to capture (10 photos needed), then enter your name when prompted. The mean embedding is saved to `data/register/<name>.npy`.

> If you change the detection or cropping pipeline, delete the old `.npy` files and re-enrol. Templates made with a different pipeline will not match reliably.

### Run live verification

```bash
python src/verify.py
```

A webcam window opens. The largest detected face is matched against all enrolled templates and labelled `UNLOCKED: <name> (score)` or `UNKNOWN (score)`. Press `Q` to quit.

### Evaluate on LFW

```bash
python src/evaluate.py
```

Computes cosine similarity for the LFW pairs, finds the threshold with the best accuracy and reports ROC-AUC.

---

## Key Technical Details

- **Identical preprocessing for training and inference.** The model is trained on MTCNN crops (detected box + 14 px margin, resized to 112×112 with PIL bilinear). `face_utils.py` reproduces exactly the same crop on live webcam frames using `mtcnn.detect()` + `extract_face()`, so the model sees the same kind of input at enrolment and verification as it did in training. An earlier version used a Haar cascade at inference, which framed faces differently and reduced accuracy.
- **Low-light detection.** Before detection, each frame's brightness channel (L in LAB colour space) is enhanced with CLAHE, and MTCNN's stage thresholds are lowered to `[0.5, 0.6, 0.6]`. The *enhanced* frame is used only to find the face; the crop is taken from the *original* frame, because the model was trained on unenhanced images.
- **Colour order.** OpenCV frames are BGR; training images were loaded as RGB with PIL. `face_utils.py` converts BGR → RGB before cropping.
- **Normalisation.** Pixels are scaled to [-1, 1] via `(x / 255 − 0.5) / 0.5`, matching `get_transforms(train=False)`.

---

## Requirements

- Python 3.11
- CUDA-capable GPU recommended (tested on RTX 3050 6 GB)
- See `requirements.txt` for the full pinned dependency list

Key packages: `torch 2.2.2`, `torchvision 0.17.2`, `facenet-pytorch 2.6.0`, `numpy 1.26.4`, `Pillow 10.2.0`, `opencv-python 4.8.1`, `scikit-learn`, `tqdm`

---

## Limitations

- **No liveness detection:** a printed photo of an enrolled person may be accepted
- Recognition accuracy still degrades in poor lighting, since the model was not trained with low-light augmentation (detection is more robust thanks to MTCNN + CLAHE)
- Faces are cropped but not landmark-aligned, so strong head tilt reduces accuracy
- Threshold (0.60) was tuned empirically; the optimal value depends on your use case and should be re-tuned after pipeline changes

---

## References

- [ArcFace: Additive Angular Margin Loss (Deng et al., 2019)](https://arxiv.org/abs/1801.07698)
- [Deep Residual Learning for Image Recognition (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- [FaceNet (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832)
- [MTCNN (Zhang et al., 2016)](https://arxiv.org/abs/1604.02878)
- [CASIA-WebFace (Yi et al., 2014)](https://arxiv.org/abs/1411.7923)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
