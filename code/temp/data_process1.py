import pandas as pd
import numpy as np
import os
import joblib
from glob import glob
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------------- 全局配置 --------------------------
# 数据集根路径
BASE_PATH = r"F:\毕业设计\数据集level1"
LINK_TABLE_DIR = os.path.join(BASE_PATH, "关联表")
MEAS_TABLE_DIR = os.path.join(BASE_PATH, "量测场景")

# 输出根路径
OUTPUT_ROOT = "./MTAD"
TRAIN_DIR = os.path.join(OUTPUT_ROOT, "训练集")
VAL_DIR = os.path.join(OUTPUT_ROOT, "验证集")
TEST_DIR = os.path.join(OUTPUT_ROOT, "测试集")

# 航迹处理参数
SEQ_LEN = 100
MIN_TRAJ_LEN = 5
STEP_SIZE = 20  # 滑动步长，充分利用航迹
FEATURE_COLS = ["lat", "lon", "vel", "cou"]

# 处理前N个场景
MAX_SCENE_NUM = 3000

# 随机种子
np.random.seed(42)

# -------------------------- 步骤1：创建输出文件夹 --------------------------
for dir_path in [OUTPUT_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)
print(f"✅ 输出文件夹已创建：{OUTPUT_ROOT}")

# -------------------------- 步骤2：获取并匹配文件列表 --------------------------
# 1. 获取所有关联表文件，按编号排序
link_file_list = glob(os.path.join(LINK_TABLE_DIR, "关联结果-*.csv"))
link_file_list.sort(key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]))
# 限制前3000个
link_file_list = link_file_list[:MAX_SCENE_NUM]

# 2. 构建文件匹配字典：{场景ID: (关联表路径, 量测表路径)}
scene_file_map = {}
for link_path in link_file_list:
    # 提取场景ID
    scene_id = int(os.path.basename(link_path).split("-")[1].split(".")[0])
    # 匹配对应的量测表
    meas_path = os.path.join(MEAS_TABLE_DIR, f"场景-{scene_id}.csv")
    if os.path.exists(meas_path):
        scene_file_map[scene_id] = (link_path, meas_path)
    else:
        print(f"⚠️  场景{scene_id}的量测表不存在，跳过")

valid_scene_ids = sorted(list(scene_file_map.keys()))
print(f"✅ 共匹配到有效场景：{len(valid_scene_ids)}个")

# -------------------------- 步骤3：按7:2:1划分训练/验证/测试集（场景级） --------------------------
# 先划分训练集(70%)和剩余(30%)
train_ids, temp_ids = train_test_split(valid_scene_ids, test_size=0.3, random_state=42)
# 再把剩余划分为验证集(20%总)和测试集(10%总)
val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)

# 保存场景划分清单
split_info = {
    "scene_id": valid_scene_ids,
    "split": ["训练集" if i in train_ids else "验证集" if i in val_ids else "测试集" for i in valid_scene_ids]
}
pd.DataFrame(split_info).to_csv(os.path.join(OUTPUT_ROOT, "场景划分清单.csv"), index=False)
print(f"✅ 场景划分完成：训练集{len(train_ids)}个，验证集{len(val_ids)}个，测试集{len(test_ids)}个")

# -------------------------- 步骤4：预拟合全局归一化器（用所有场景数据） --------------------------
print("正在预拟合全局特征归一化器...")
all_features = []
for scene_id in valid_scene_ids[:100]:  # 为了效率，用前100个场景拟合足够
    _, meas_path = scene_file_map[scene_id]
    try:
        df_meas = pd.read_csv(meas_path)
        df_meas.columns = df_meas.columns.str.strip()
        features = df_meas[FEATURE_COLS].fillna(0).values
        all_features.append(features)
    except Exception as e:
        continue

all_features = np.concatenate(all_features, axis=0)
global_scaler = MinMaxScaler()
global_scaler.fit(all_features)
# 保存全局归一化器
joblib.dump(global_scaler, os.path.join(OUTPUT_ROOT, "全局特征归一化器.save"))
print("✅ 全局归一化器拟合完成并保存")

# -------------------------- 步骤5：循环处理每个场景并保存 --------------------------
def process_single_scene(scene_id, link_path, meas_path, save_dir):
    """处理单个场景并保存到指定目录"""
    try:
        # 1. 读取数据
        df_link = pd.read_csv(link_path)
        df_meas = pd.read_csv(meas_path)
        
        # 2. 修复列名
        df_link.columns = df_link.columns.str.strip()
        df_meas.columns = df_meas.columns.str.strip()
        
        # 3. 数据类型转换
        df_link[["batch", "source"]] = df_link[["batch", "source"]].apply(pd.to_numeric, errors="coerce")
        df_meas[["batch", "source"]] = df_meas[["batch", "source"]].apply(pd.to_numeric, errors="coerce")
        
        # 4. 关联两个表
        df_meas_with_mmsi = pd.merge(
            df_meas,
            df_link[["mmsi", "batch", "source"]].dropna(),
            on=["batch", "source"],
            how="inner"
        )
        
        if len(df_meas_with_mmsi) == 0:
            return False
        
        # 5. 提取航迹段
        scene_trajectories = []
        scene_metadata = []
        
        for (mmsi, batch, source), group in df_meas_with_mmsi.groupby(["mmsi", "batch", "source"]):
            group_sorted = group.sort_values("time").reset_index(drop=True)
            traj_len = len(group_sorted)
            if traj_len < MIN_TRAJ_LEN:
                continue
            
            # 滑动窗口切分
            for i in range(0, traj_len - SEQ_LEN + 1, STEP_SIZE):
                traj_segment = group_sorted.loc[i:i+SEQ_LEN-1, FEATURE_COLS].fillna(0).values
                # 全局归一化
                traj_segment_norm = global_scaler.transform(traj_segment)
                scene_trajectories.append(traj_segment_norm)
                # 记录元数据
                scene_metadata.append({
                    "traj_idx": len(scene_trajectories)-1,
                    "mmsi": mmsi,
                    "batch": batch,
                    "source": source
                })
        
        if len(scene_trajectories) == 0:
            return False
        
        # 6. 保存文件
        scene_trajectories = np.array(scene_trajectories, dtype=np.float32)
        np.save(os.path.join(save_dir, f"scene_{scene_id}_traj_X.npy"), scene_trajectories)
        pd.DataFrame(scene_metadata).to_csv(os.path.join(save_dir, f"scene_{scene_id}_metadata.csv"), index=False)
        
        return True
    except Exception as e:
        print(f"❌ 场景{scene_id}处理出错：{str(e)}")
        return False

# 开始批量处理
print("="*60)
print("开始批量处理场景...")
print("="*60)

count_train, count_val, count_test = 0, 0, 0

for scene_id in valid_scene_ids:
    link_path, meas_path = scene_file_map[scene_id]
    
    # 判断保存目录
    if scene_id in train_ids:
        save_dir = TRAIN_DIR
        count_train += 1
    elif scene_id in val_ids:
        save_dir = VAL_DIR
        count_val += 1
    else:
        save_dir = TEST_DIR
        count_test += 1
    
    # 处理并保存
    success = process_single_scene(scene_id, link_path, meas_path, save_dir)
    
    if success and (count_train + count_val + count_test) % 100 == 0:
        print(f"已处理 {count_train + count_val + count_test}/{len(valid_scene_ids)} 个场景...")

print("="*60)
print(f"✅ 全部处理完成！")
print(f"训练集保存：{count_train} 个场景 -> {TRAIN_DIR}")
print(f"验证集保存：{count_val} 个场景 -> {VAL_DIR}")
print(f"测试集保存：{count_test} 个场景 -> {TEST_DIR}")
print(f"场景划分清单：{os.path.join(OUTPUT_ROOT, '场景划分清单.csv')}")
print(f"全局归一化器：{os.path.join(OUTPUT_ROOT, '全局特征归一化器.save')}")
print("="*60)