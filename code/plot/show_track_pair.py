import numpy as np
import matplotlib.pyplot as plt
import os
import random
import platform
from matplotlib.font_manager import FontProperties, findfont

# ===================== Configuration =====================
# Track pair data root directory
DATA_ROOT = "/home/yangcq/track_association/data/track_pairs/"
# Output figure path
OUTPUT_FIGURE_PATH = "/home/yangcq/track_association/output/figures/random_track_pair_visualization.png"
# Feature columns (keep consistent with before)
FEATURE_COLS = ["lat", "lon", "vel", "cou"]
# Random seed (set to None for different random results each time)
RANDOM_SEED = None
# =====================================================================

def set_font():
    """Set font for better display (no Chinese needed)"""
    # Use default sans-serif font, works on all platforms
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load processed track pair data and scaler parameters"""
    # Load training set (most samples, easy to find positive/negative pairs)
    track1 = np.load(os.path.join(DATA_ROOT, "track1_train.npy"))
    track2 = np.load(os.path.join(DATA_ROOT, "track2_train.npy"))
    labels = np.load(os.path.join(DATA_ROOT, "labels_train.npy"))
    
    # Load scaler parameters
    scaler_mean = np.load(os.path.join(DATA_ROOT, "scaler_mean.npy"))
    scaler_scale = np.load(os.path.join(DATA_ROOT, "scaler_scale.npy"))
    
    return track1, track2, labels, scaler_mean, scaler_scale

def inverse_normalize(track, scaler_mean, scaler_scale):
    """Inverse normalization to recover real latitude and longitude"""
    # track shape: (max_len, 4)
    return track * scaler_scale + scaler_mean

def get_valid_points_normalized_space(track):
    """
    [CRITICAL] Filter padding zeros in NORMALIZED SPACE (BEFORE inverse normalization)
    Padding zeros here are true [0,0,0,0], easy to identify
    """
    zero_mask = np.all(track == 0, axis=1)
    if np.any(zero_mask):
        first_zero_idx = np.argmax(zero_mask)
        return track[:first_zero_idx]
    else:
        return track

def plot_single_pair(ax, track1, track2, label, title):
    """Plot a pair of trajectories on a single subplot"""
    # Extract latitude and longitude (first two dimensions)
    t1_lat = track1[:, 0]
    t1_lon = track1[:, 1]
    t2_lat = track2[:, 0]
    t2_lon = track2[:, 1]
    
    # Plot source 9001
    ax.plot(t1_lon, t1_lat, 'b-', linewidth=2, alpha=0.8, label='Source 9001')
    ax.scatter(t1_lon, t1_lat, color='blue', s=15, alpha=0.7)
    ax.scatter(t1_lon[0], t1_lat[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(t1_lon[-1], t1_lat[-1], color='red', s=100, marker='x', label='End')
    
    # Plot source 9002
    ax.plot(t2_lon, t2_lat, 'r-', linewidth=2, alpha=0.8, label='Source 9002')
    ax.scatter(t2_lon, t2_lat, color='red', s=15, alpha=0.7)
    ax.scatter(t2_lon[0], t2_lat[0], color='green', s=100, marker='o')
    ax.scatter(t2_lon[-1], t2_lat[-1], color='red', s=100, marker='x')
    
    # Set plot properties
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.legend(fontsize=12)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

if __name__ == "__main__":
    set_font()
    print("="*70)
    print("Track Pair Visualization")
    print("="*70)

    # Set random seed
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # 1. Load data
    print("\n[1/3] Loading data...")
    track1_data, track2_data, labels_data, scaler_mean, scaler_scale = load_data()
    print(f" Data loaded successfully")
    print(f"   Total samples: {len(labels_data)}")
    print(f"   Positive samples: {int(np.sum(labels_data))}")
    print(f"   Negative samples: {int(len(labels_data) - np.sum(labels_data))}")

    # 2. Randomly sample pairs
    print("\n[2/3] Randomly sampling pairs...")
    
    # Sample positive pair (label=1)
    pos_indices = np.where(labels_data == 1)[0]
    if len(pos_indices) == 0:
        print(" No positive samples found")
        exit()
    pos_idx = random.choice(pos_indices)
    
    # Sample negative pair (label=0)
    neg_indices = np.where(labels_data == 0)[0]
    if len(neg_indices) == 0:
        print(" No negative samples found")
        exit()
    neg_idx = random.choice(neg_indices)
    
    print(f" Sampled positive pair index: {pos_idx}")
    print(f" Sampled negative pair index: {neg_idx}")

    # 3. [FIXED] Filter padding zeros FIRST in normalized space, THEN inverse normalize
    print("\n[3/3] Filtering padding zeros and visualizing...")
    
    # Process positive pair
    pos_t1 = track1_data[pos_idx]
    pos_t2 = track2_data[pos_idx]
    # [CRITICAL 1] First filter in normalized space
    pos_t1_valid_norm = get_valid_points_normalized_space(pos_t1)
    pos_t2_valid_norm = get_valid_points_normalized_space(pos_t2)
    # [CRITICAL 2] Only inverse normalize the valid points
    pos_t1_valid = inverse_normalize(pos_t1_valid_norm, scaler_mean, scaler_scale)
    pos_t2_valid = inverse_normalize(pos_t2_valid_norm, scaler_mean, scaler_scale)
    
    # Process negative pair
    neg_t1 = track1_data[neg_idx]
    neg_t2 = track2_data[neg_idx]
    # [CRITICAL 1] First filter in normalized space
    neg_t1_valid_norm = get_valid_points_normalized_space(neg_t1)
    neg_t2_valid_norm = get_valid_points_normalized_space(neg_t2)
    # [CRITICAL 2] Only inverse normalize the valid points
    neg_t1_valid = inverse_normalize(neg_t1_valid_norm, scaler_mean, scaler_scale)
    neg_t2_valid = inverse_normalize(neg_t2_valid_norm, scaler_mean, scaler_scale)

    # 4. Plot visualization
    plt.figure(figsize=(16, 7))
    
    # Left subplot: Positive pair
    ax1 = plt.subplot(1, 2, 1)
    plot_single_pair(ax1, pos_t1_valid, pos_t2_valid, 1, 
                     f"Label=1 (Same Target)\nSource 9001: {len(pos_t1_valid)} pts | Source 9002: {len(pos_t2_valid)} pts")
    
    # Right subplot: Negative pair
    ax2 = plt.subplot(1, 2, 2)
    plot_single_pair(ax2, neg_t1_valid, neg_t2_valid, 0, 
                     f"Label=0 (Different Targets)\nSource 9001: {len(neg_t1_valid)} pts | Source 9002: {len(neg_t2_valid)} pts")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print(f" Visualization complete!")
    print(f" Figure saved to: {OUTPUT_FIGURE_PATH}")
    print(f" Left: Positive pair (Label=1), two trajectories are similar")
    print(f" Right: Negative pair (Label=0), two trajectories are different")
    print("="*70)