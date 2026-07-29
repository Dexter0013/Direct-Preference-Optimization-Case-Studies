# Direct Preference Optimization for Temporal Window Selection in Object Detection

A DPO-trained gating policy that decides **when** to run an object detector on video, rather than running it on every frame. Trained on the EPFL IXMAS dataset, validated on real-world footage (UCF101, custom uploads).

## Problem

Running an object detector on every frame of a video is wasteful, many frames are redundant, blurry, or simply not worth the compute. This project trains a lightweight policy via **Direct Preference Optimization (DPO)** to look at a short temporal window of frames and decide: *is this moment worth running the detector on, or can we skip it and reuse the last result?*

The policy is a small Transformer scorer (~1M parameters) sitting in front of a frozen, much heavier detector (RF-DETR-Nano). It never touches the detector's weights, it only learns when to call it.

## How it works

```
Video → Temporal windows (8 frames, stride 4) → DINOv2-Small embedding (384-dim) → WindowSelectorPolicy → score → score > threshold? → run detector : reuse last detection
```

The policy is trained via DPO on preference pairs generated from the detector's own confidence scores: for each video, overlapping windows are scored by RF-DETR-Nano's max person-detection confidence, and pairs are formed wherever two windows differ meaningfully in score (`winner` vs. `loser`). The policy learns to rank windows the same way, without ever seeing ground-truth labels.

## Architecture

| Component | Role | Trained? |
|---|---|---|
| **DINOv2-Small** | Converts each frame/window into a 384-dim embedding | Frozen |
| **RF-DETR-Nano** | Person detector; source of preference labels during training, and the actual detector at inference | Frozen |
| **`WindowSelectorPolicy`** | `Linear(384→256)` → 2-layer `TransformerEncoder` (4 heads, ff=512) → `Linear(256→1)` | **Trained via DPO** |

```python
class WindowSelectorPolicy(nn.Module):
    def __init__(self, dim=384, hidden=256, heads=4, ff=512, layers=2):
        self.proj = nn.Linear(dim, hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=ff, batch_first=True),
            num_layers=layers
        )
        self.score_head = nn.Linear(hidden, 1)
```

**DPO loss** (reference policy is implicit uniform, cancels out):
```
L = -log σ(β · (score(winner) - score(loser))),  β = 0.5
```

## Pipeline stages

| Stage | Description |
|---|---|
| 0 | Load EPFL IXMAS dataset, filter to Camera 3 (single fixed view) |
| 1 | Split streams into train/val (80/20) **before** pair generation — guarantees zero leakage |
| 2 | Extract frames via ffmpeg pipe (raw `bgr24`, `48×64`) |
| 3 | Slice into overlapping 8-frame windows, score with RF-DETR-Nano, generate preference pairs |
| 4 | Embed each window with DINOv2-Small (mean-pooled across frames) |
| 5 | Train `WindowSelectorPolicy` via DPO |
| 6 | Deploy: gate detector calls on local video files (chunked or frame-by-frame) |
| 7 | *(Optional)* Hysteresis smoothing of detect/skip decisions to reduce flicker |

# Model Outputs with dataset threshold=5.09 for Object Detection pipeline
- Output 1
<video src="https://github.com/user-attachments/assets/a2dc78e1-976d-47af-8cd1-71c85e1f813c" controls width="600"></video>
- Output 2
<video src="https://github.com/user-attachments/assets/1538d75b-91a4-4028-bd84-1bd69583ab49" controls width="600"></video>
## Results
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/cc692e6e-f845-4611-93de-ae99a1aa579d" />


Evaluated on a **stream-level train/val split**

### Ranking quality

| Metric | Train | Val |
|---|---|---|
| Pairwise accuracy | 99.2% | **100%** |
| ROC AUC | 0.956 | 0.991 |
| PR AUC | 0.967 | 0.992 |
| Winner score mean | 10.41 | 11.82 |
| Loser score mean | -4.12 | -3.56 |

Val accuracy ≥ train accuracy (`gap = -0.008`) — no overfitting.

### Threshold calibration (best F1)

| Threshold | Precision | Recall | F1 | Detector call rate |
|---|---|---|---|---|
| **5.10** | 0.951 | 0.954 | 0.953 | 50.1% |

<img width="1289" height="440" alt="image" src="https://github.com/user-attachments/assets/1e3997f4-2fa0-481d-98cf-b5111aa47e6d" />

### Compute efficiency — recall at matched budget

| Gating method | Mean recall @ budget |
|---|---|
| **DPO policy** | **45.8%** |
| Uniform stride | 37.3% |
| Random | 25.1% |

At the same compute budget, the trained policy catches **~1.8× more relevant windows than random gating**.

### Speed

| | ms / window |
|---|---|
| Policy | 0.68 |
| Detector | 30.1 |

The detector is ~29.1x more expensive then the Policy.

## Repository structure

```
Model/
  Generalized_DPO(pytorch_model).ipynb   # end-to-end pipeline notebook
```

## Requirements

```bash
pip install rfdetr supervision tqdm torch transformers scikit-learn opencv-python-headless
apt-get install -y ffmpeg
```

## Quickstart

```python
# Load trained policy
policy = WindowSelectorPolicy()
policy.load_state_dict(torch.load("dpo_policy_clean.pt"))
policy.eval()

# Run on a local video file
results = process_local_video_chunked(
    "your_video.mp4",
    policy, detector, dinov2_embedder,
    threshold=5.098,
    use_generic_extractor=True   # use for any non-EPFL-format source
)
```

## Known limitations

- Trained entirely on EPFL IXMAS (lab-controlled, single-person, fixed camera) — the score distribution shifts on more naturalistic footage, though the policy still discriminates sensibly (see *Generalization* above).
- The `.avi`/ffmpeg raw-frame extractor is specific to IXMAS's uncompressed format; any other video source should use the generic `cv2`-based extractor.
- Detector confidence (not ground-truth boxes) is used as the preference signal — the policy learns to match the detector's own notion of a "good window," not an independent quality measure.

## License
**© Deepraj Singha 2026**
