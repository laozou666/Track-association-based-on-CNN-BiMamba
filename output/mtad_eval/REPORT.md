# MTAD Dataset Evaluation Report

Source: 崔亚奇等, 基于全球 AIS 的多源航迹关联数据集 (MTAD) V2, Science Data Bank, 2025
Root:   `/home/yangcq/track_association/data/raw`

## 1. Association table overview
- **n_scenes**: 5000
- **tracks_total**: 369167
- **tracks_per_scene_mean**: 73.833400
- **tracks_per_scene_std**: 152.851691
- **tracks_per_scene_min**: 10
- **tracks_per_scene_max**: 2273
- **n_9001_total**: 204552
- **n_9002_total**: 164615
- **shared_mmsi_per_scene_mean**: 21.933200
- **pos_pairs_total**: 246956
- **neg_pairs_max_total**: 35300177
- **pos_over_total**: 0.006947
- **mmsi_suffix_top**: {'0': 246013, '1': 123154}
- **source_counter**: {9001: 204552, 9002: 164615}

## 2. Measurement statistics (sampled scenes)
- **track_length_steps**: n=25296, mean=283.176, std=195.632, min=3.000, p50=240.000, p95=712.000, max=720.000
- **track_duration_sec**: n=25296, mean=3623.381, std=2093.094, min=31.765, p50=3184.676, p95=7130.000, max=7200.000
- **inter_sample_dt_sec**: n=7137935, mean=12.841, std=4.519, min=10.013, p50=10.015, p95=20.071, max=26.667
- **lat**: n=7163231, mean=19.959, std=0.384, min=-1.247, p50=19.964, p95=20.479, max=26.877
- **lon**: n=7163231, mean=130.978, std=0.512, min=123.233, p50=130.980, p95=131.783, max=136.648
- **vel**: n=7163231, mean=57.310, std=199.311, min=0.005, p50=47.135, p95=114.224, max=63257.007
- **cou**: n=7163231, mean=179.847, std=104.282, min=0.000, p50=179.739, p95=342.109, max=360.000

## 3. Noise analysis (measurement vs ground truth)
- **position_error_m**: n=1045216, mean=1013.677, std=1599.227, p50=276.813, p95=4049.849, max=116030.432
- **velocity_diff_knot**: n=1045216, mean=53.720, std=116.053, p50=44.105, p95=111.791, max=27848.984
- **course_diff_deg**: n=1045216, mean=81.596, std=52.087, p50=76.726, p95=169.099, max=180.000

## 4. Comparison with `final_dataset` used for training
```
{
  "final_dataset_present": true,
  "train": {
    "shape": [
      67748,
      350,
      4
    ],
    "max_len_configured": 350,
    "pos_ratio": 0.6790015941430005,
    "length_median": 78.0,
    "length_max": 350,
    "length_min": 21
  },
  "val": {
    "shape": [
      9678,
      350,
      4
    ],
    "max_len_configured": 350,
    "pos_ratio": 0.6789625955775986,
    "length_median": 78.0,
    "length_max": 350,
    "length_min": 21
  },
  "test": {
    "shape": [
      19357,
      350,
      4
    ],
    "max_len_configured": 350,
    "pos_ratio": 0.6790308415560262,
    "length_median": 78.0,
    "length_max": 350,
    "length_min": 21
  }
}
```

## 5. Usage suggestion
- Positive pairs for training: all (batch, 9001) x (batch, 9002) tuples that share the same MMSI core inside a scene.
- Negative pairs: sample cross-source tuples with different MMSI core from the same scene (recommended to roughly balance or to use class-weighting like the current pipeline).
- Features per step: `lat, lon, vel, cou` — same 4-D layout as the current model input, so weights transfer in principle.
- Sequences are variable length with much larger max length than the current 350-step cap: either re-pad to a new max, downsample, or window the measurements before feeding CNN-BiMamba.
- Ground-truth interpolation from `真实场景` allows noise-error visualization and physical sanity checks.
