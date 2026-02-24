#!/bin/bash
# 控制变量验证脚本
# 分别测试视觉增强和动力学随机化对模型鲁棒性的影响

set -e

# 配置参数
CKPT_DIR="/Volumes/Elements/act/training_results/ft_robust_20260220_145121"
TASK_NAME="sim_transfer_cube_scripted"
POLICY_CLASS="ACT"
BATCH_SIZE=8
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
DIM_FEEDFORWARD=3200
ROBUST_SEED=2026
ROBUST_NUM_ROLLOUTS=50

echo "=========================================="
echo "控制变量验证实验"
echo "模型目录: $CKPT_DIR"
echo "=========================================="
echo ""

# 实验1: 仅视觉扰动测试
echo "[实验1] 运行仅视觉扰动测试 (vision_only)..."
python imitate_episodes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class "$POLICY_CLASS" \
    --task_name "$TASK_NAME" \
    --batch_size $BATCH_SIZE \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --dim_feedforward $DIM_FEEDFORWARD \
    --robust_eval \
    --robust_mode vision_only \
    --robust_suite transfer_vision_v2 \
    --robust_seed $ROBUST_SEED \
    --robust_num_rollouts $ROBUST_NUM_ROLLOUTS \
    --robust_save_json

echo ""
echo "[实验1] 完成！结果保存在: $CKPT_DIR/robust_vision_summary.json"
echo ""

# 实验2: 仅动力学随机化测试
echo "[实验2] 运行仅动力学随机化测试 (dynamics_only)..."
python imitate_episodes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class "$POLICY_CLASS" \
    --task_name "$TASK_NAME" \
    --batch_size $BATCH_SIZE \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --dim_feedforward $DIM_FEEDFORWARD \
    --robust_eval \
    --robust_mode dynamics_only \
    --robust_suite transfer_dynamics_v1 \
    --robust_seed $ROBUST_SEED \
    --robust_num_rollouts $ROBUST_NUM_ROLLOUTS \
    --robust_save_json

echo ""
echo "[实验2] 完成！结果保存在: $CKPT_DIR/robust_dynamics_summary.json"
echo ""

# 生成对比报告
echo "[对比分析] 生成综合对比报告..."
python robust_compare.py \
    --ckpt_dir "$CKPT_DIR" \
    --task_name "$TASK_NAME" \
    --policy_class "$POLICY_CLASS" \
    --batch_size $BATCH_SIZE \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --dim_feedforward $DIM_FEEDFORWARD \
    --robust_seed $ROBUST_SEED \
    --robust_num_rollouts $ROBUST_NUM_ROLLOUTS \
    --skip_runs

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结果文件:"
echo "  - 视觉测试: $CKPT_DIR/robust_vision_summary.json"
echo "  - 动力学测试: $CKPT_DIR/robust_dynamics_summary.json"
echo "  - 对比报告: $CKPT_DIR/robust_compare_report.txt"
echo "  - 对比数据: $CKPT_DIR/robust_compare_summary.json"
echo "=========================================="
