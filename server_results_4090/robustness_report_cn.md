# 鲁棒性评估总结报告

## 1. 执行背景
- 评估时间: 2026-02-16 (结果文件最新时间戳: 21:16)
- 评估任务: `sim_transfer_cube_scripted`
- 模型目录: `server_results_4090`
- 评估方式: 单因素分离评估
- 动力学干扰: `dynamics_only` (`transfer_dynamics_v1`)
- 视觉干扰: `vision_only` (`transfer_vision_v1`)
- 每类评估样本数: 50 rollouts

## 2. 数据源与一致性校验
本报告基于以下文件:
- `server_results_4090/robust_compare_summary.json`
- `server_results_4090/robust_compare_report.txt`
- `server_results_4090/robust_dynamics_summary.json`
- `server_results_4090/robust_vision_summary.json`
- `server_results_4090/result_policy_best_dynamics_only.txt`
- `server_results_4090/result_policy_best_vision_only.txt`

一致性检查结果:
- `dynamics_only` success_rate / avg_return: JSON 与 TXT 完全一致
- `vision_only` success_rate / avg_return: JSON 与 TXT 完全一致
- 差值均为 `0.0`

## 3. 核心指标对比

| 指标 | dynamics_only | vision_only | 差值 (dyn - vis) |
|---|---:|---:|---:|
| Success Rate | 0.82 | 0.42 | +0.40 |
| Average Return | 550.36 | 282.84 | +267.52 |
| Avg Highest Reward | 3.46 | 2.12 | +1.34 |

结论:
- 当前模型在动力学扰动下保持较高性能。
- 当前模型在视觉扰动下性能明显下降。
- 主要短板在视觉鲁棒性，不在动力学泛化。

## 4. 细分干扰分析

### 4.1 动力学干扰分层
- `moderate` (n=4): success_rate=0.75, avg_return=498.25
- `severe` (n=46): success_rate=0.8261, avg_return=554.8913

观察:
- 在本次采样中，`severe` 组未劣于 `moderate` 组。
- 说明当前参数采样定义下，策略对动力学变化整体较稳。

### 4.2 视觉干扰分类型
按 success_rate 从低到高排序:
1. `gaussian_noise` (n=10): success_rate=0.00, avg_return=9.2
2. `occlusion` (n=8): success_rate=0.25, avg_return=118.5
3. `jpeg_compression` (n=13): success_rate=0.3846, avg_return=278.6154
4. `motion_blur` (n=9): success_rate=0.6667, avg_return=446.0
5. `brightness_contrast` (n=10): success_rate=0.80, avg_return=546.6

观察:
- 最薄弱项是高斯噪声，成功率为 0。
- 遮挡和压缩也有显著影响。
- 亮度/对比度变化影响相对最小。

## 5. 结论
- 本次结果明确表明: 模型对视觉干扰更敏感。
- 在 transfer 任务中，动力学扰动并非首要风险。
- 视觉感知链路是当前鲁棒性提升的主攻方向。

## 6. 建议的下一步
1. 先补一组无干扰基线 (`--robust_eval` 关闭) 做同目录对照，量化真实衰减比例。
2. 针对 `gaussian_noise` 做强度分桶评估 (低/中/高噪声)，找出性能崩溃阈值。
3. 视觉增广优先级建议: 高斯噪声 > 遮挡 > JPEG 压缩。
4. 若继续做报告对比，建议固定随机种子并保存每轮配置，减少跨轮波动。

## 7. 附: 关键文件
- 综合报告: `server_results_4090/robust_compare_report.txt`
- 综合结构化数据: `server_results_4090/robust_compare_summary.json`
- 本文档: `server_results_4090/robustness_report_cn.md`
