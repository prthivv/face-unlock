# Face Unlock

A real-time facial recognition and biometric authentication system trained from scratch using **ResNet50** and **ArcFace loss** on the **CASIA-WebFace** dataset. Built as a university machine learning project in two weeks.

---

## Overview

This project implements a complete face verification pipeline similar to smartphone Face ID:

- **Train** a deep embedding model to produce discriminative 512-D face representations
- **Enrol** a new user by capturing 10 webcam photos and storing their mean embedding
- **Verify** in real time — a live camera feed is compared against enrolled templates using cosine similarity

The architecture and training setup mirrors published research: ResNet50 backbone + ArcFace loss + CASIA-WebFace dataset is the standard combination used across most face recognition papers.

---

## Demo

| Enrolled user | Unknown person |
|:---:|:---:|
| `UNLOCKED: Matthew (0.73)` | `UNKNOWN (0.31)` |

---

## Architecture

```
Webcam Frame
     │
     ▼
OpenCV Face Detection (Haar Cascade)
     │
     ▼
112×112 Crop + Normalise [-1, 1]
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
- All images preprocessed with MTCNN → aligned 112×112 crops
- 489,409 images retained after face detection filtering

**Evaluation:** [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- 6,000 pre-defined face pairs (same / different person)
- Standard benchmark used in all published face recognition papers

---

## Project Structure

```
face-unlock/
├── data/
│   ├── raw/                  # Downloaded datasets (CASIA-WebFace, LFW)
│   ├── processed/            # MTCNN-aligned 112×112 crops
│   └── register/             # Enrolled face templates (.npy)
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset class (reads casia-webface.txt)
│   │   └── preprocess.py     # MTCNN alignment pipeline
│   ├── models/
│   │   ├── backbone.py       # ResNet50 feature extractor
│   │   └── arcface.py        # ArcFace loss module
│   ├── train.py              # Training loop (AMP, StepLR, checkpointing)
│   ├── enroll.py             # Webcam enrolment script
│   ├── verify.py             # Real-time verification script
│   └── evaluate.py           # LFW benchmark evaluation
├── checkpoints/              # Saved model weights
├── requirements.txt
└── README.md
```

---

## Setup for training

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/face-unlock.git
cd face-unlock
conda create -n faceunlock python=3.10
conda activate faceunlock
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download and preprocess CASIA-WebFace

Download [CASIA-WebFace from Kaggle](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) and extract to `data/raw/casia-webface/`.

Run MTCNN alignment (takes ~4 hours on CPU — run overnight):

```bash
cd src
python data/preprocess.py
```

---

## Training

```bash
cd src
python train.py
```

Checkpoints are saved to `checkpoints/` every 5 epochs. Training ran for 25 epochs on an RTX 3050 6 GB (~5 hours), reaching a final ArcFace loss of ~1.65.

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
cd src
python enroll.py
```

Press `SPACE` to capture 10 photos. Enter your name when prompted. The mean embedding is saved to `data/register/<name>.npy`.

### Run live verification

```bash
cd src
python verify.py
```

A webcam window opens. Detected faces are matched against all enrolled templates. Press `Q` to quit.

### Evaluate on LFW

```bash
cd src
python evaluate.py
```

Computes cosine similarity for all LFW pairs and finds the optimal verification threshold.

---

## Key Technical Details

## Requirements

- Python 3.10
- CUDA-capable GPU (tested on RTX 3050 6 GB)
- See `requirements.txt` for full dependency list

Key packages: `torch 2.2`, `torchvision 0.17`, `facenet-pytorch 2.6`, `opencv-python 4.8`, `scikit-learn`, `tqdm`

---

## Limitations

- **No liveness detection** — a printed photo of an enrolled person may be accepted
- Performance degrades under severe lighting changes or heavy occlusion
- Threshold (0.60) was tuned empirically; optimal value depends on your use case

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
