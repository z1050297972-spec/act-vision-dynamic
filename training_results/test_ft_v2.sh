#!/bin/bash
# 测试二次微调模型的鲁棒性

CKPT_DIR="training_results/ft_robust_20260223_103058"
TASK_NAME="sim_transfer_cube_scripted"

echo "=========================================="
echo "测试二次微调模型"
echo "模型目录: $CKPT_DIR"
echo "=========================================="

# 检查 policy_best.ckpt 是否存在
if [ ! -f "$CKPT_DIR/policy_best.ckpt" ]; then
    echo "警告: policy_best.ckpt 不存在，使用 policy_epoch_1900_seed_0.ckpt"
    cp "$CKPT_DIR/policy_epoch_1900_seed_0.ckpt" "$CKPT_DIR/policy_best.ckpt"
fi

# 测试 1: 仅视觉扰动
echo ""
echo "[测试 1/3] 仅视觉扰动..."
python imitate_episodes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --task_name "$TASK_NAME" \
    --policy_class ACT \
    --batch_size 8 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --robust_eval \
    --robust_mode vision_only \
    --robust_suite transfer_vision_v2 \
    --robust_seed 2026 \
    --robust_num_rollouts 50 \
    --robust_save_json

# 测试 2: 仅动力学随机化
echo ""
echo "[测试 2/3] 仅动力学随机化..."
python imitate_episodes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --task_name "$TASK_NAME" \
    --policy_class ACT \
    --batch_size 8 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --robust_eval \
    --robust_mode dynamics_only \
    --robust_suite transfer_dynamics_v1 \
    --robust_seed 2026 \
    --robust_num_rollouts 50 \
    --robust_save_json

# 测试 3: 视觉+动力学联合
echo ""
echo "[测试 3/3] 视觉+动力学联合..."
python imitate_episodes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --task_name "$TASK_NAME" \
    --policy_class ACT \
    --batch_size 8 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --robust_eval \
    --robust_mode vision_dynamics \
    --robust_suite transfer_vision_v2 \
    --robust_seed 2026 \
    --robust_num_rollouts 50 \
    --robust_save_json

# 生成对比报告
echo ""
echo "[生成报告] 对比分析..."
python robust_compare.py \
    --ckpt_dir "$CKPT_DIR" \
    --task_name "$TASK_NAME" \
    --policy_class ACT \
    --batch_size 8 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --robust_seed 2026 \
    --robust_num_rollouts 50 \
    --skip_runs

echo ""
echo "=========================================="
echo "测试完成！"
echo "结果文件:"
echo "  - $CKPT_DIR/robust_vision_summary.json"
echo "  - $CKPT_DIR/robust_dynamics_summary.json"
echo "  - $CKPT_DIR/robust_vision_dynamics_summary.json"
echo "  - $CKPT_DIR/robust_compare_report.txt"
echo "=========================================="
