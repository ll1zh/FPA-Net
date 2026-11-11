> **FPA-Net: Frequency and Position-Aware Deep Network for Low-Light Image Enhancement in HVI Color Space**

---

## 1. Environment

| item | value |
|---|---|
| Python | 3.10.16 |
| PyTorch | 2.1.2 |
| CUDA | 11.8 |
| GPU | NVIDIA RTX 4090 (a single GPU is sufficient for training and inference) |

```bash
pip install -r requirements.txt
```

---

## 2. Datasets

```
datasets
├── LOLdataset                 # LOLv1
│   ├── our485/{low,high}      # 485 training pairs
│   └── eval15/{low,high}      # 15 test pairs
├── LOLv2
│   ├── Real_captured
│   │   ├── Train/{Low,Normal}  # 689 training pairs
│   │   └── Test/{Low,Normal}   # 100 test pairs
│   └── Synthetic
│       ├── Train/{Low,Normal}  # 900 training pairs
│       └── Test/{Low,Normal}   # 100 test pairs
└── FiveK                      # MIT-5K, expert C as reference
    ├── train/{input,target}   # 4500 pairs
    └── test/{input,target}    # 500 pairs
```

---

## 3. Training

```bash
# LOLv1
python train.py --dataset lol_v1

# LOLv2-Real
python train.py --dataset lolv2_real

# LOLv2-Synthetic
python train.py --dataset lolv2_syn

# MIT-5K
python train.py --dataset fivek
```

The default configuration is the one used in the paper: 1000 epochs, batch size 8,
256×256 patches, Adam with a cosine schedule (3 warm-up epochs), learning rate
1e-4 → 1e-7, and the loss weights λ=1.0, λ₁=1.0, λ_s=0.5, λ_e=50, λ_p=1e-2.
Any of them can be overridden on the command line, e.g. `--batchSize 8 --seed 0`.

Outputs:

* checkpoints → `weights/<dataset>/epoch_<N>.pth`
* metrics log → `results/training/metrics_<dataset>_<timestamp>.md`
* a sample prediction of the last batch → `results/training/`


---

## 4. Evaluation

```bash
# 1) run the model over the test split; the checkpoint is a command-line argument
python eval.py --dataset lol_v1     --ckpt weight/lolv1_best_psnr_lpips.pth
python eval.py --dataset lolv2_real --ckpt weight/lolv2_best_psnr.pth --alpha 0.8

# 2) compute the metrics of the enhanced images
python measure.py --dataset lol_v1
python measure.py --dataset lolv2_real
```

