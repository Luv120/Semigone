# 20-Hour Sprint Plan — Team Semigone

## The Math

| Activity | Hours | Notes |
|----------|-------|-------|
| **Block 1:** Scaffold + Data + Eval Script | 3h | Everything that touches the filesystem |
| **Block 2:** NAFNet + Losses + Training Loop | 4h | The model code |
| **Block 3:** Training Run #1 | 2.5h | ⚡ Colab runs unattended |
| **↳ Parallel:** Slides + README | 2.5h | You work on these WHILE training runs |
| **Block 4:** Evaluate + Fix + Retrain if needed | 3h | Look at results, iterate |
| **Block 5:** Polish eval script + GitHub repo | 3h | Fresh-env test |
| **Block 6:** Final outputs + submission | 2h | Generate restored test images, upload |
| **Buffer** | 2.5h | Things will break |
| **Total** | **20h** | |

> [!IMPORTANT]
> The trick is **Block 3**: training takes ~2.5h on the T4 but requires zero input from you. That's 2.5 free hours for slides and README. Effective coding time is ~15h.

---

## What's IN (Scored)

- NAFNet-Tiny (0.48M params)
- Charbonnier + SSIM loss (dropped FFT loss — marginal gain, extra code)
- KLA paired data loader with geometric augmentations (flip + rotate90) (.npy) dataset:
```kla_semicon_ps1_dataset/
│
├── Train/
│   ├── Ground_Truth/ (or Train_HR/)   # Clean, high-resolution SEM target images
│   └── Degraded/     (or Train_LR/)   # Downsampled, noisy SEM input images
│
└── Test/
    └── Test_NoisyLR/                  # Degraded test images (no GT provided)
        ├── In_Distribution/           # Wafer patterns similar to training set
        └── Out_of_Distribution/       # Unseen structural geometries / defect types
```

- Standalone `evaluate.py` with auto-resolution detection
- `train.py` with checkpoint resume
- `requirements.txt`, `README.md`
- Restored test outputs folder
- 9-slide PDF

## What's OUT (Not building)

- ~~XGBoost defect detector~~
- ~~QC verifier~~
- ~~Classical baselines (Lee, SRAD)~~
- ~~FFT loss term~~ (Charbonnier + SSIM is sufficient)
- ~~MixUp / CutMix~~ (nice but not worth the code time)
- ~~Gradio demo~~ (optional, only if buffer time remains)
- ~~Synthetic degradation pipeline~~ (you have KLA's real data)
- ~~TTA self-ensemble~~ (we'll add a simple version only if time permits)

---

## Block 1: Scaffold + Data + Eval Script (Hours 0–3)

**This is where I do most of the heavy lifting with you.**

### Files created this block:

```
semicon-restore/
├── configs/default.yaml
├── data/dataset.py              # KLA pair loader + augmentations
├── models/                      # empty, filled in Block 2
├── evaluate.py                  # ★ Skeleton that runs end-to-end
├── requirements.txt             # Known dependencies upfront
└── README.md                    # Skeleton
```

### `evaluate.py` — skeleton-first

```python
# Runs NOW with a dummy model (identity function)
# python evaluate.py --input_dir /path/to/test/ --output_dir /path/to/output/
#
# Later: swap dummy model for trained NAFNet
# The contract never changes. KLA's H100 runs this file AS-IS.
```

Write it so it works immediately with a bicubic-upsample placeholder. Once the model is trained, you swap one line.

### `dataset.py` — keep it simple

```python
# 1. Scan input_dir and target_dir, match filenames
# 2. Load as grayscale float32
# 3. Normalize: input /= input.max() (preserves speckle overflow)
#                target /= 255.0 (ground truth is clean, standard range)
# 4. Random crop: (patch_lr, patch_hr) at 2× ratio
# 5. Random flip + rotate90 (applied identically to both)
# 6. Return (lr_tensor, hr_tensor)
```

### What you do during Block 1:
- Upload KLA dataset to Google Drive
- Explore the data: check resolutions, intensity ranges, file formats
- Verify the dataset structure (how are pairs organized? naming convention?)

---

## Block 2: NAFNet + Losses + Training Loop (Hours 3–7)

### Files created this block:

```
├── models/
│   ├── nafnet.py                # NAFNet backbone
│   ├── nafnet_sr.py             # + PixelShuffle SR head + global residual
│   └── losses.py                # Charbonnier + SSIM (2 terms only)
├── train.py                     # Full training script with resume
```

### `losses.py` — two terms, not four

$$\mathcal{L} = \underbrace{\sqrt{(x-y)^2 + \epsilon^2}}_{\text{Charbonnier}} + 0.1 \cdot \underbrace{(1 - \text{SSIM}(x, y))}_{\text{SSIM}}$$

That's it. Charbonnier handles pixel fidelity, SSIM handles structure. Two loss functions, ~30 lines of code.

### `train.py` — with Colab survival built in

```python
# Key features:
# --resume /path/to/checkpoint.pt     (survive Colab disconnects)
# --save_dir /content/drive/MyDrive/  (checkpoints on Drive, not Colab local)
# --max_iters 15000
# --val_interval 500                  (validate + print metrics)
# --save_interval 1000                (checkpoint to Drive)
#
# Prints: [iter 500/15000] loss=0.0234 | val_psnr=28.4 | val_ssim=0.812 | ETA: 2h14m
```

### Training config:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard, stable |
| Learning rate | 1e-3 | NAFNet default |
| Weight decay | 1e-4 | Mild regularization |
| Scheduler | Cosine annealing | Smooth LR decay |
| Warmup | 500 iters | Prevent early instability |
| Batch size | 8 | Fits T4 16GB comfortably |
| LR patch size | 64×64 | HR = 128×128 at 2× |
| Max iterations | 15,000 | ~2.5h on T4 |
| Validation | Every 500 iters | Track progress |
| Checkpoint | Every 1,000 iters | Survive disconnects |

---

## Block 3: Training + Parallel Slides (Hours 7–10)

### On Colab (unattended):

```bash
!python train.py \
    --config configs/default.yaml \
    --data_dir /content/drive/MyDrive/kla_data/ \
    --save_dir /content/drive/MyDrive/semicon-checkpoints/ \
    --max_iters 15000
```

**Keep the Colab tab active** (pin it, wiggle mouse occasionally, or use a keep-alive extension).

### While training runs — write slides (Google Slides → PDF):

| Slide | Time | What to write |
|-------|------|---------------|
| 1. Team Details | 5 min | Semigone, members, college, contacts |
| 2. Problem Statement | 10 min | 1 paragraph: noisy low-res SEM images → need fast restoration for yield |
| 3. Approach | 15 min | NAFNet-Tiny, joint denoise+SR, single forward pass, no cascaded pipeline |
| 4. Solution Detail | 25 min | Architecture diagram, loss formula, training strategy |
| 5. Innovation | 15 min | (a) Intensity-aware normalization for speckle overflow, (b) Efficient architecture (0.48M params, <10ms), (c) Global residual learning |
| 6. Results | *Fill after Block 4* | Leave placeholder — fill with actual numbers after training |
| 7. Tech Stack | 10 min | PyTorch, Colab T4, training time, model size, inference speed |
| 8. GitHub + Video | 5 min | Repo link |
| 9. References | 5 min | NAFNet paper citation |

**Also during this time:** Write the README.md (setup instructions, 3-command quickstart).

---

## Block 4: Evaluate + Iterate (Hours 10–13)

### Check training results:

```python
# Load best checkpoint
# Run on validation split
# Compute PSNR, SSIM, LPIPS
# Generate before/after comparison images
```

### Decision tree:

```
PSNR ≥ 28 dB AND SSIM ≥ 0.80?
  ├── YES → Move to Block 5. You're in good shape.
  └── NO → Diagnose:
           ├── Loss still decreasing? → Train longer (extend to 25k iters)
           ├── Loss plateaued low but metrics bad? → Check data loading bug
           ├── Loss exploded? → Reduce LR to 5e-4, retrain
           └── Overfitting (train good, val bad)? → Add dropout=0.05, retrain
```

### If results are decent, use remaining time to:
1. Add simple TTA (just horizontal flip + average — 2× cost, easy to implement, ~0.2–0.4 dB free)
2. Fill in Slide 6 (Results) with actual numbers and before/after images

---

## Block 5: Polish + Fresh-Env Test (Hours 13–16)

### Finalize evaluate.py:

```python
# Swap the dummy bicubic model for the trained NAFNet
# Test the EXACT command KLA will run:
python evaluate.py --input_dir test_images/ --output_dir restored_outputs/

# Verify:
# ✓ All output images exist
# ✓ Output resolution = 2× input resolution
# ✓ Output values in [0, 255] uint8 (or [0, 1] float — match KLA's expected format)
# ✓ Same filenames as input
# ✓ No errors, no warnings, no prompts
```

### Fresh-environment test (critical):

```bash
# On Colab, create a new runtime:
!git clone https://github.com/yourname/semicon-restore.git
%cd semicon-restore
!pip install -r requirements.txt
!python evaluate.py --input_dir /content/test_dummy/ --output_dir /content/out/
# If this fails, FIX IT before anything else.
```

### GitHub repo final structure:

```
semicon-restore/
├── configs/default.yaml
├── data/dataset.py
├── models/
│   ├── nafnet.py
│   ├── nafnet_sr.py
│   └── losses.py
├── weights/
│   └── nafnet_tiny.pt          # ~2MB
├── outputs/                    # Restored test images
│   ├── test_001.png
│   ├── test_002.png
│   └── ...
├── evaluate.py                 # ★ THE file
├── train.py
├── requirements.txt
└── README.md
```

---

## Block 6: Final Submission (Hours 16–18)

1. **Generate restored test outputs**: Run evaluate.py on KLA's test set, save to `outputs/`
2. **Push to GitHub**: Final commit, verify repo is public
3. **Export slides as PDF**: `Semigone_KLA_PS01.pdf`
4. **Upload to i4C portal**: PDF + GitHub link
5. **Sanity check**: Clone your own repo one more time, run evaluate.py, verify it works

### Buffer (Hours 18–20)

Things that might eat buffer time:
- Colab disconnect during training → resume from checkpoint
- Data loading bugs (unexpected file format, naming mismatch)
- CUDA version mismatch in requirements.txt
- Slides taking longer than expected
- Model weights too large for GitHub (use Git LFS or Google Drive link)

---

## Colab Session Survival Guide

| Risk | Mitigation |
|------|-----------|
| Idle disconnect (~90 min) | Pin tab, interact periodically, or use browser keep-alive extension |
| Session kill (~12h) | Checkpoints every 1k iters to Google Drive |
| GPU not available | Try reconnecting; worst case, train on CPU overnight (slower but works) |
| Drive mount drops | Re-mount at start of each session: `drive.mount('/content/drive')` |
| RAM OOM | Reduce batch_size to 4; use `del` + `gc.collect()` aggressively |

---

## The 20-Hour Clock

```
 ┌──────────────────────────────────────────────────────────────┐
 │ Hr 0─────3─────7──────10──────13──────16───18───20          │
 │    [SCAFFOLD] [MODEL]  [TRAIN]  [EVAL]  [POLISH] [SUBMIT]  │
 │    data+eval  nafnet   ║slides║  iterate  fresh   upload    │
 │    script     losses   ║readme║  fix      env     done!     │
 │    explore    train.py ║      ║  TTA?     test             │
 │                        ╚══════╝                             │
 │                        ↑ parallel work while GPU trains     │
 └──────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **The one rule:** If it's hour 16 and your evaluate.py doesn't run clean on a fresh env, **drop everything else and fix it.** An unscored submission with a beautiful model is worth zero. A scored submission with a mediocre model at least competes.
