MalVEIT/
│
├── train.py
├── test.py
├── config.py
│
├── datasets/
│   └── malimg.py
│
├── models/
│   ├── patch_embed.py          ← Patch Partition
│   ├── masking.py              ← Random Masking
│   ├── mask_token.py           ← Learnable Mask Token
│   ├── swin_encoder.py         ← Swin Transformer
│   ├── bottleneck.py           ← μ, σ Projection
│   ├── reparameterize.py
│   ├── decoder.py              ← VAE Decoder
│   ├── classifier.py
│   ├── losses.py
│   └── malveit.py              ← Complete Architecture
│
├── utils/
│   ├── metrics.py
│   ├── gradcam.py
│   └── visualization.py
│
└── checkpoints/
