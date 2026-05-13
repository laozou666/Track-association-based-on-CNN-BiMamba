# Length-Stratified Test Evaluation

## Test Set Information

| Split | Length Range | Mean Length | N | Pos | Neg |
|---|---|:---:|:---:|:---:|:---:|
| Short | [21, 79] | 50.5 | 9,836 | 7,010 | 2,826 |
| Medium | [80, 180] | 115.8 | 7,239 | 4,717 | 2,522 |
| Long | [181, 350] | 284.0 | 2,282 | 1,417 | 865 |
| **Total** | [21, 350] | 102.4 | **19,357** | 13,144 | 6,213 |

## Accuracy Summary (%)

| Model | Short | Medium | Long | Weighted Overall |
|---|:---:|:---:|:---:|:---:|
| CNN-Mamba tuned (Ours) | 92.87 | 92.26 | 92.77 | **92.63** |
| CNN-BiLSTM | 91.92 | 91.74 | 93.69 | **92.06** |
| CNN-LSTM | 82.16 | 80.08 | 81.42 | **81.29** |

## Full Metrics per Split

### Short Test Set  (length [21, 79], N = 9,836)

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| CNN-Mamba tuned (Ours) | 92.87% | 92.60% | 97.82% | 95.14% |
| CNN-BiLSTM | 91.92% | 93.30% | 95.52% | 94.40% |
| CNN-LSTM | 82.16% | 85.73% | 89.93% | 87.78% |

### Medium Test Set  (length [80, 180], N = 7,239)

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| CNN-Mamba tuned (Ours) | 92.26% | 92.07% | 96.44% | 94.20% |
| CNN-BiLSTM | 91.74% | 91.89% | 95.78% | 93.79% |
| CNN-LSTM | 80.08% | 83.80% | 86.07% | 84.92% |

### Long Test Set  (length [181, 350], N = 2,282)

| Model | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| CNN-Mamba tuned (Ours) | 92.77% | 93.29% | 95.20% | 94.24% |
| CNN-BiLSTM | 93.69% | 93.04% | 97.11% | 95.03% |
| CNN-LSTM | 81.42% | 84.27% | 86.17% | 85.21% |

## Accuracy Advantage: CNN-Mamba tuned  vs  CNN-BiLSTM

| Split | CNN-Mamba tuned | CNN-BiLSTM | Gap (pp) |
|---|:---:|:---:|:---:|
| Short | 92.87% | 91.92% | **+0.96** |
| Medium | 92.26% | 91.74% | **+0.52** |
| Long | 92.77% | 93.69% | **-0.92** |
