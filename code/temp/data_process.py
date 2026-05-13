import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

# -------------------------- 全局配置（可根据你的需求修改） --------------------------
# 固定航迹序列长度（和你的CNN+LSTM模型输入长度保持一致）
SEQ_LEN = 100
# 正负样本比例 1:3
POS_NEG_RATIO = 1/3
# 过滤过短的航迹（少于MIN_TRAJ_LEN个点的航迹直接丢弃）
MIN_TRAJ_LEN = 5
# 随机种子，保证结果可复现
np.random.seed(42)

# -------------------------- 步骤1：读取数据 + 过滤关联表 --------------------------
# 1.1 读取两个CSV文件（替换为你的文件路径）
df_link = pd.read_csv("F:\毕业设计\数据集level1\关联表\关联结果-3.csv")  # 图1的量测表
df_meas = pd.read_csv("F:\毕业设计\数据集level1\量测场景\场景-3.csv")  # 图2的关联表


# 1.2 关联表预处理：只保留核心字段，剔除无效值
df_link = df_link[["mmsi", "source", "batch"]].dropna().reset_index(drop=True)
# 把batch转为数值型，避免字符串和数字匹配失败
df_link["batch"] = pd.to_numeric(df_link["batch"], errors="coerce")
df_meas["batch"] = pd.to_numeric(df_meas["batch"], errors="coerce")

# 1.3 【严格匹配你的需求1】剔除一个mmsi只对应一个batch的数据
# 按mmsi分组，统计每个mmsi对应的batch数量
mmsi_batch_count = df_link.groupby("mmsi")["batch"].nunique().reset_index()
# 只保留batch数量≥2的mmsi（至少2条航迹才能组对）
valid_mmsi = mmsi_batch_count[mmsi_batch_count["batch"] >= 2]["mmsi"].tolist()
# 过滤后的关联表
df_link_filtered = df_link[df_link["mmsi"].isin(valid_mmsi)].reset_index(drop=True)
print(f"步骤1完成：过滤后剩余有效mmsi数量 {len(valid_mmsi)}")

# -------------------------- 步骤2：打通航迹与mmsi的关联，提取完整航迹 --------------------------
# 2.1 给量测表的航迹点打上mmsi标签（通过batch+source匹配关联表）
df_meas_with_mmsi = pd.merge(
    df_meas,
    df_link_filtered,
    on=["batch", "source"],
    how="inner"  # 只保留关联表中存在的有效航迹
)

# 2.2 提取每条完整航迹：同mmsi+同batch+同source 对应一条航迹，按时间排序
trajectory_dict = {}  # key: 航迹唯一ID，value: 航迹特征序列
traj_info = []        # 记录每条航迹对应的mmsi、source、batch

# 按航迹唯一标识分组
for (mmsi, batch, source), group in df_meas_with_mmsi.groupby(["mmsi", "batch", "source"]):
    # 按时间戳排序，保证航迹时序正确
    group_sorted = group.sort_values("time").reset_index(drop=True)
    # 过滤过短的航迹
    if len(group_sorted) < MIN_TRAJ_LEN:
        continue
    # 提取航迹核心特征（和你的模型输入特征对应：lat, lon, vel, cou）
    traj_feature = group_sorted[["lat", "lon", "vel", "cou"]].values
    # 生成航迹唯一ID
    traj_id = f"{mmsi}_{batch}_{source}"
    trajectory_dict[traj_id] = traj_feature
    traj_info.append({
        "traj_id": traj_id,
        "mmsi": mmsi,
        "batch": batch,
        "source": source
    })

# 转为DataFrame方便后续处理
df_traj_info = pd.DataFrame(traj_info)
print(f"步骤2完成：提取有效航迹总数 {len(trajectory_dict)}")

# -------------------------- 步骤3：生成航迹对（正样本+负样本，1:3比例） --------------------------
# 3.1 【严格匹配你的需求2、3】生成正样本对：同一个mmsi的不同航迹两两配对
positive_pairs = []
# 按mmsi分组，对每个mmsi的航迹两两组合
for mmsi, group in df_traj_info.groupby("mmsi"):
    traj_ids = group["traj_id"].tolist()
    # 同一个mmsi的航迹两两组合，不重复
    for traj_a, traj_b in combinations(traj_ids, 2):
        positive_pairs.append([traj_a, traj_b, 1])  # 标签1=正样本（同一艘船）

# 限制正样本数量，避免样本过多（可选，根据你的需求调整）
# positive_pairs = positive_pairs[:5000]
pos_count = len(positive_pairs)
neg_count = int(pos_count / POS_NEG_RATIO)  # 负样本数量=正样本的3倍
print(f"正样本对数量：{pos_count}，需生成负样本对数量：{neg_count}")

# 3.2 生成负样本对：不同mmsi的航迹随机配对
negative_pairs = []
all_traj_ids = list(trajectory_dict.keys())
all_mmsi_list = df_traj_info["mmsi"].unique().tolist()

# 循环生成足够的负样本
while len(negative_pairs) < neg_count:
    # 随机选两个不同的mmsi
    mmsi_a, mmsi_b = np.random.choice(all_mmsi_list, 2, replace=False)
    # 从两个mmsi中各随机选一条航迹
    traj_a = df_traj_info[df_traj_info["mmsi"] == mmsi_a]["traj_id"].sample(1).values[0]
    traj_b = df_traj_info[df_traj_info["mmsi"] == mmsi_b]["traj_id"].sample(1).values[0]
    # 避免重复配对
    if [traj_a, traj_b, 0] not in negative_pairs and [traj_b, traj_a, 0] not in negative_pairs:
        negative_pairs.append([traj_a, traj_b, 0])  # 标签0=负样本（不同船）

# 3.3 合并正负样本对
all_pairs = positive_pairs + negative_pairs
np.random.shuffle(all_pairs)  # 打乱样本顺序
print(f"步骤3完成：总航迹对数量 {len(all_pairs)}")

# -------------------------- 步骤4：航迹序列对齐+归一化，适配模型输入 --------------------------
# 4.1 特征归一化：把所有航迹的特征归一化到[0,1]区间，提升模型训练效果
all_features = np.concatenate(list(trajectory_dict.values()), axis=0)
scaler = MinMaxScaler()
scaler.fit(all_features)

# 4.2 序列对齐函数：长的截断，短的补零，统一为SEQ_LEN长度
def pad_or_truncate_traj(traj, seq_len=SEQ_LEN):
    traj_len = len(traj)
    if traj_len >= seq_len:
        # 长航迹：截断前seq_len个点
        return traj[:seq_len]
    else:
        # 短航迹：末尾补零
        pad_width = ((0, seq_len - traj_len), (0, 0))
        return np.pad(traj, pad_width, mode="constant", constant_values=0)

# 4.3 生成最终的模型输入数据
X = []  # 航迹对数据，shape: [样本数, 2, SEQ_LEN, 特征维度]
y = []  # 标签，shape: [样本数]

for traj_a_id, traj_b_id, label in all_pairs:
    # 取出两条航迹的原始特征
    traj_a = trajectory_dict[traj_a_id]
    traj_b = trajectory_dict[traj_b_id]
    # 归一化
    traj_a_norm = scaler.transform(traj_a)
    traj_b_norm = scaler.transform(traj_b)
    # 序列对齐
    traj_a_padded = pad_or_truncate_traj(traj_a_norm)
    traj_b_padded = pad_or_truncate_traj(traj_b_norm)
    # 加入数据集
    X.append([traj_a_padded, traj_b_padded])
    y.append(label)

# 转为numpy数组，可直接喂入PyTorch模型
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# -------------------------- 保存处理好的数据 --------------------------
np.save("航迹对特征3_X.npy", X)
np.save("航迹对标签3_y.npy", y)
# 保存归一化器，后续推理用
import joblib
joblib.dump(scaler, "特征归一化器.save")

print(f"数据处理全部完成！")
print(f"最终数据集形状：X={X.shape}, y={y.shape}")
print(f"文件已保存：航迹对特征_X.npy、航迹对标签_y.npy")