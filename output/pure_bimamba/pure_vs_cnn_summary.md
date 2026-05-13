# Pure BiMamba vs CNN-BiMamba (tuned)

Fixed: `n_mamba=1, dropout=0.4`

| Variant | Params | Val Acc | Best Epoch |
|---------|--------|---------|------------|
| Pure BiMamba (n_cnn=0) | 152,641 | 91.50% | 14 |
| CNN-BiMamba tuned (n_cnn=4) | 265,921 | 92.38% | 23 |
