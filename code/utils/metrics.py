import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

def calculate_all_metrics(labels, preds, verbose=True):
    """
    计算二分类所有核心指标
    :param labels: 真实标签，numpy数组
    :param preds: 预测标签，numpy数组
    :param verbose: 是否打印结果
    :return: 指标字典
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    if verbose:
        print("="*40)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("="*40)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
def get_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 混淆矩阵 (numpy array)
    """
    return confusion_matrix(y_true, y_pred)

def save_metrics_to_txt(metrics_dict, save_path):
    """
    把评估指标保存到文本文件
    :param metrics_dict: 指标字典
    :param save_path: 保存路径
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("测试集评估结果\n")
        f.write("="*60 + "\n")
        for name, value in metrics_dict.items():
            f.write(f"{name:10s}: {value:.4f}\n")
    print(f"✅ 评估指标已保存到：{save_path}")