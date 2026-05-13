#!/bin/bash
# Train all 8 models in parallel across 4 GPUs (2 models per GPU)
# Run: bash code/train_all.sh
# Each model is pinned to a specific GPU via CUDA_VISIBLE_DEVICES

cd /home/yangcq/track_association/code

EPOCHS=30
BATCH=64

echo "========================================"
echo " Launching 8 models across 4 GPUs"
echo "========================================"

# GPU 0 — ann, bilstm
CUDA_VISIBLE_DEVICES=0 conda run -n track_association python3 train.py \
    --model ann     --epochs $EPOCHS --batch_size $BATCH > /tmp/log_ann.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 conda run -n track_association python3 train.py \
    --model bilstm  --epochs $EPOCHS --batch_size $BATCH > /tmp/log_bilstm.txt 2>&1 &

# GPU 1 — bigru, lstm
CUDA_VISIBLE_DEVICES=1 conda run -n track_association python3 train.py \
    --model bigru   --epochs $EPOCHS --batch_size $BATCH > /tmp/log_bigru.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 conda run -n track_association python3 train.py \
    --model lstm    --epochs $EPOCHS --batch_size $BATCH > /tmp/log_lstm.txt 2>&1 &

# GPU 2 — cnn, cnn_lstm
CUDA_VISIBLE_DEVICES=2 conda run -n track_association python3 train.py \
    --model cnn      --epochs $EPOCHS --batch_size $BATCH > /tmp/log_cnn.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 conda run -n track_association python3 train.py \
    --model cnn_lstm --epochs $EPOCHS --batch_size $BATCH > /tmp/log_cnn_lstm.txt 2>&1 &

# GPU 3 — cnn_bilstm, cnn_mamba (mamba 独占一张卡)
CUDA_VISIBLE_DEVICES=3 conda run -n track_association python3 train.py \
    --model cnn_bilstm --epochs $EPOCHS --batch_size $BATCH > /tmp/log_cnn_bilstm.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 conda run -n track_association python3 train.py \
    --model cnn_mamba  --epochs $EPOCHS --batch_size $BATCH > /tmp/log_cnn_mamba.txt 2>&1 &

echo "All 8 training jobs launched. Waiting for completion..."

# 等所有后台任务完成
wait

echo ""
echo "========================================"
echo " All models finished. Summary:"
echo "========================================"
for MODEL in ann bilstm bigru lstm cnn cnn_lstm cnn_bilstm cnn_mamba; do
    LOG="/tmp/log_${MODEL}.txt"
    # 提取最佳验证指标
    BEST=$(grep "最佳模型已更新" $LOG | tail -1)
    LAST_ACC=$(grep "验证 Loss" $LOG | tail -1)
    echo "[$MODEL]  $LAST_ACC"
done

echo ""
echo "========================================"
echo " Running comparison evaluation..."
echo "========================================"
conda run -n track_association python3 comparision.py
