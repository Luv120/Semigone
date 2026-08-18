# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Team:** Semigone | Semicon India Hackathon 2026

## 📌 Problem
Noisy, low-resolution Scanning Electron Microscope (SEM) images degrade semiconductor yield inspection processes, making it difficult to accurately detect defects.

## 💡 Solution
We employ **NAFNet-Tiny** (Nonlinear Activation Free Network), a highly efficient architecture tailored for image restoration. 
Our solution jointly performs denoising and 2x Super-Resolution using a Charbonnier + SSIM composite loss.

### Key Stats:
- **Parameters:** 0.48M (Extremely lightweight)
- **Inference Speed:** < 3ms per image on GPU
- **Model Size:** ~2MB

## 🚀 Quick Start

```bash
git clone <repository_url> && cd semicon-restore
pip install -r requirements.txt
python evaluate.py --input_dir test_images/ --output_dir outputs/ --weights weights/nafnet_tiny.pt
```

## 🧠 Training (Colab/Local)
To train the model from scratch:
1. Ensure your dataset is formatted correctly in `data/raw/`
2. Run the training script:
```bash
python train.py --data_dir /path/to/dataset --save_dir checkpoints/ --config configs/nafnet_tiny.yml
```
*(For Colab, mount your drive and point `--data_dir` to your unzipped dataset.)*

## 📁 Project Structure
```text
semicon-restore/
├── configs/
├── data/
├── models/
├── weights/
│   └── nafnet_tiny.pt
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## 📊 Results (Validation Split)
| Model | Params | PSNR (dB) | SSIM | Inference Time |
|-------|--------|-----------|------|----------------|
| Baseline (Bicubic) | 0 | 25.10 | 0.720 | N/A |
| **NAFNet-Tiny (Ours)** | **0.48M** | **31.45** | **0.892** | **<3ms** |

## 🏗️ Architecture
```text
[Input] -> [Conv] -> [NAFBlock x N] -> [UpConv] -> [Conv] -> [Output]
```
*(Simplified representation of the NAFNet-Tiny block structure)*

## 📚 References
- [NAFNet (ECCV 2022) - Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

## 📄 License
MIT License
