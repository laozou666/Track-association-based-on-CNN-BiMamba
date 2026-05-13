# Mamba Direction Ablation

Fixed config: `n_cnn=4, n_mamba=1, dropout=0.4`

| Direction | Params | Val Acc | Val Loss | Best Epoch |
|-----------|--------|---------|----------|------------|
| fwd | 227,009 | 90.52% | 0.1664 | 23 |
| bwd | 227,009 | 91.55% | 0.1664 | 22 |
| bi | 265,921 | 92.38% | 0.1600 | 23 |
