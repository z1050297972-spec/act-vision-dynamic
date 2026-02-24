# ACT 项目文件结构说明

本文档详细说明项目中每个文件和目录的作用，帮助理解整个代码库的组织结构。

## 📁 项目目录树

```
act/
├── .git/                           # Git 版本控制目录
├── .vscode/                        # VS Code 编辑器配置
├── assets/                         # MuJoCo 仿真资源文件
│   ├── *.xml                       # 机器人和场景定义文件
│   └── *.stl                       # 3D 模型文件
├── data/                           # 训练数据集目录
│   └── episode_*.hdf5              # HDF5 格式的轨迹数据
├── detr/                           # DETR (Detection Transformer) 模型实现
│   ├── models/                     # 模型架构定义
│   │   ├── backbone.py             # ResNet 骨干网络
│   │   ├── detr_vae.py             # DETR-VAE 主模型
│   │   ├── transformer.py          # Transformer 编码器/解码器
│   │   └── position_encoding.py   # 位置编码
│   └── util/                       # 工具函数
├── training_results/               # 训练输出目录
│   └── ft_robust_*/                # 微调实验结果
├── server_results_4090/            # 服务器训练结果
├── constants.py                    # 全局常量定义
├── utils.py                        # 数据加载和增强工具
├── policy.py                       # 策略网络定义
├── scripted_policy.py              # 脚本化专家策略
├── sim_env.py                      # 基础仿真环境
├── ee_sim_env.py                   # 末端执行器仿真环境
├── record_sim_episodes.py          # 数据采集脚本
├── imitate_episodes.py             # 训练和评估主脚本
├── visualize_episodes.py          # 数据可视化工具
├── robustness.py                   # 鲁棒性测试框架
├── robust_compare.py               # 鲁棒性对比分析
├── run_vision_robust_matrix.py     # 视觉鲁棒性矩阵实验
├── run_controlled_validation.sh    # 控制变量验证 Shell 脚本
├── conda_env.yaml                  # Conda 环境配置
├── README.md                       # 项目说明文档
└── WORKTREE.md                     # 本文档

```

---

## 📄 核心文件详解

### 🔧 配置和常量

#### `constants.py`
定义项目的全局常量和配置参数。

**主要内容：**
- `SIM_TASK_CONFIGS`: 任务配置字典（数据集路径、episode 长度、相机名称）
- `DT`: 仿真时间步长 (0.02s)
- `JOINT_NAMES`: 机器人关节名称列表
- `START_ARM_POSE`: 机器人初始姿态
- `XML_DIR`: MuJoCo XML 文件路径
- 夹爪位置/关节限制和归一化函数

**作用：** 集中管理所有硬编码的参数，便于修改和维护。

#### `conda_env.yaml`
Conda 环境配置文件，定义项目依赖。

**主要依赖：**
- PyTorch (深度学习框架)
- MuJoCo (物理仿真)
- h5py (数据存储)
- OpenCV (图像处理)

---

### 🤖 策略和模型

#### `policy.py`
定义策略网络类，是模型的高层封装。

**主要类：**
- `ACTPolicy`: Action Chunking Transformer 策略
  - 使用 CVAE (条件变分自编码器) 架构
  - 训练时计算 L1 损失和 KL 散度
  - 推理时从先验分布采样动作序列
- `CNNMLPPolicy`: CNN+MLP 基线策略
  - 简单的卷积神经网络 + 多层感知机
  - 单步动作预测

**作用：** 提供统一的策略接口，封装训练和推理逻辑。

#### `scripted_policy.py`
专家策略实现，用于生成演示数据。

**主要类：**
- `BasePolicy`: 策略基类
  - 轨迹插值功能
  - 噪声注入选项
- `PickAndTransferPolicy`: 抓取和传递任务策略
  - 定义双臂协作轨迹
  - 右臂抓取 → 传递 → 左臂接收
- `InsertionPolicy`: 插入任务策略
  - 双臂协作插入轨迹
  - 精确的位置和姿态控制

**作用：** 生成高质量的演示数据，用于模仿学习。

---

### 🎮 仿真环境

#### `sim_env.py`
基础仿真环境实现。

**功能：**
- 封装 MuJoCo 物理引擎
- 提供标准的 `reset()` 和 `step()` 接口
- 处理关节空间控制
- 渲染和观测管理

#### `ee_sim_env.py`
末端执行器（End-Effector）仿真环境。

**功能：**
- 基于 `sim_env.py` 扩展
- 支持笛卡尔空间控制（位置 + 姿态）
- 逆运动学求解
- 双臂协作任务支持

**作用：** 提供更直观的末端控制接口，简化策略设计。

---

### 📊 数据处理

#### `utils.py`
数据加载、增强和工具函数集合。

**主要功能：**

1. **数据增强配置**
   - `_default_train_aug_config()`: 默认视觉增强参数
   - `_build_train_aug_config()`: 构建和验证增强配置
   - `_default_action_aug_config()`: 默认动作增强参数

2. **数据集类**
   - `EpisodicDataset`: PyTorch 数据集类
     - 加载 HDF5 格式的 episode 数据
     - 应用视觉增强（高斯噪声、运动模糊、遮挡等）
     - 应用动作增强（噪声、延迟）
     - 数据归一化

3. **数据加载**
   - `load_data()`: 创建训练/验证数据加载器
   - `get_norm_stats()`: 计算归一化统计量

4. **环境工具**
   - `sample_box_pose()`: 随机采样物体位置
   - `sample_insertion_pose()`: 随机采样插入任务位置

**作用：** 核心数据管道，支持灵活的数据增强策略。

---

### 🏋️ 训练和评估

#### `imitate_episodes.py`
训练和评估的主脚本。

**主要功能：**

1. **训练模式**
   - 加载数据集
   - 构建策略网络
   - 训练循环（前向传播、反向传播、优化）
   - 保存检查点和训练曲线

2. **评估模式**
   - 加载训练好的模型
   - 在仿真环境中 rollout
   - 计算成功率和平均回报
   - 支持鲁棒性测试

3. **鲁棒性评估**
   - 视觉扰动测试
   - 动力学随机化测试
   - 联合测试

**命令行参数：**
- `--task_name`: 任务名称
- `--ckpt_dir`: 检查点目录
- `--policy_class`: 策略类型 (ACT/CNNMLP)
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--eval`: 评估模式
- `--robust_eval`: 鲁棒性评估
- `--train_visual_aug`: 启用视觉增强

**作用：** 项目的核心执行脚本，统一训练和评估流程。

#### `record_sim_episodes.py`
数据采集脚本。

**功能：**
- 使用脚本化策略生成演示
- 保存为 HDF5 格式
- 记录观测、动作、奖励

**作用：** 生成训练数据集。

#### `visualize_episodes.py`
数据可视化工具。

**功能：**
- 加载 HDF5 数据
- 可视化轨迹
- 生成视频
- 绘制关节位置曲线

**作用：** 检查数据质量，调试问题。

---

### 🛡️ 鲁棒性测试

#### `robustness.py`
鲁棒性测试框架核心模块。

**主要功能：**

1. **场景生成**
   - `build_robust_scenarios()`: 生成测试场景
   - 支持模式：`none`, `vision_only`, `dynamics_only`, `vision_dynamics`

2. **视觉扰动**
   - `_build_visual_cfg()`: 构建视觉扰动配置
   - 支持类型：
     - 高斯噪声 (gaussian_noise)
     - 运动模糊 (motion_blur)
     - 亮度对比度 (brightness_contrast)
     - 遮挡 (occlusion)
     - JPEG 压缩 (jpeg_compression)
   - `apply_visual_perturbation()`: 应用扰动到图像

3. **动力学随机化**
   - `_build_dynamics_cfg()`: 构建动力学配置
   - 随机化参数：
     - 物体质量
     - 摩擦系数
     - 关节阻尼
     - 执行器增益

4. **结果分析**
   - `summarize_robust_results()`: 汇总测试结果
   - 按扰动类型和严重程度分组统计

**作用：** 提供系统化的鲁棒性测试能力。

#### `robust_compare.py`
鲁棒性对比分析脚本。

**功能：**
- 自动运行三种测试模式
- 生成对比报告
- 识别哪个因素影响更大

**输出文件：**
- `robust_compare_report.txt`: 文本报告
- `robust_compare_summary.json`: JSON 数据

**作用：** 控制变量实验，量化不同因素的影响。

#### `run_vision_robust_matrix.py`
视觉鲁棒性矩阵实验脚本。

**功能：**
- 运行多组视觉增强实验 (A1-A6)
- 测试不同增强策略的效果
- 生成实验矩阵报告

**实验配置：**
- A1-A6: 不同的增强配置和课程学习策略

**作用：** 系统化探索视觉增强超参数空间。

#### `run_controlled_validation.py` / `.sh`
控制变量验证脚本（本次新增）。

**功能：**
- 分别测试视觉和动力学因素
- 自动生成对比分析
- 计算协同效应

**作用：** 明确视觉增强和动力学随机化各自的贡献。

---

### 🧠 DETR 模型

#### `detr/models/detr_vae.py`
DETR-VAE 主模型实现。

**架构：**
- ResNet 骨干网络提取图像特征
- Transformer 编码器处理特征
- VAE 解码器生成动作序列
- 支持 action chunking（预测多步动作）

#### `detr/models/transformer.py`
Transformer 编码器和解码器。

**功能：**
- 多头自注意力机制
- 前馈网络
- 层归一化

#### `detr/models/backbone.py`
ResNet 骨干网络。

**功能：**
- 使用预训练的 ResNet-18
- 提取多尺度特征

#### `detr/models/position_encoding.py`
位置编码模块。

**功能：**
- 正弦位置编码
- 为 Transformer 提供位置信息

---

## 📂 数据目录

### `data/`
存储训练数据集。

**文件格式：** HDF5 (`.hdf5`)

**数据结构：**
```
episode_0.hdf5
├── /observations
│   ├── /qpos          # 关节位置 [T, 14]
│   ├── /qvel          # 关节速度 [T, 14]
│   └── /images
│       └── /top       # 顶视图图像 [T, H, W, 3]
├── /action            # 动作序列 [T, 14]
└── attributes
    └── sim: True      # 是否为仿真数据
```

### `training_results/`
存储训练输出。

**典型内容：**
- `policy_best.ckpt`: 最佳模型检查点
- `policy_last.ckpt`: 最后一轮检查点
- `dataset_stats.pkl`: 数据集统计量
- `train_val_*.png`: 训练曲线图
- `robust_*.json`: 鲁棒性测试结果

---

## 🎯 典型工作流程

### 1. 数据采集
```bash
python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --num_episodes 50
```

### 2. 训练模型
```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir training_results/exp1 \
    --policy_class ACT \
    --num_epochs 3000 \
    --batch_size 8 \
    --train_visual_aug
```

### 3. 评估模型
```bash
python imitate_episodes.py \
    --eval \
    --ckpt_dir training_results/exp1 \
    --task_name sim_transfer_cube_scripted \
    --policy_class ACT
```

### 4. 鲁棒性测试
```bash
python run_controlled_validation.py \
    --ckpt_dir training_results/exp1
```

---

## 🔄 Git Worktree 使用指南

### 为什么使用 Worktree？

训练和评估通常需要长时间运行，使用 worktree 可以：
- 同时运行多个分支的实验
- 避免重复克隆和安装
- 隔离不同的实验修改

### 目录布局建议

```
/Volumes/Elements/act                      # 主仓库
/Volumes/Elements/act-worktrees/
├── robustness-exp                         # 鲁棒性实验分支
├── policy-refactor                        # 策略重构分支
└── vision-aug-tuning                      # 视觉增强调优分支
```

### 创建新 Worktree

```bash
cd /Volumes/Elements/act
git fetch origin
git worktree add /Volumes/Elements/act-worktrees/robustness-exp \
  -b feature/robustness-exp origin/main
```

### 日常命令

列出所有 worktree：
```bash
git worktree list
```

更新主分支：
```bash
cd /Volumes/Elements/act
git pull --ff-only
```

在 worktree 中 rebase：
```bash
cd /Volumes/Elements/act-worktrees/robustness-exp
git fetch origin
git rebase origin/main
```

### 删除 Worktree

```bash
git worktree remove /Volumes/Elements/act-worktrees/robustness-exp
git branch -d feature/robustness-exp
git worktree prune
```

### 注意事项

- 大型训练输出不要提交到 Git，使用 `.gitignore` 排除
- 所有 worktree 共享同一个 conda 环境
- 一个分支只在一个 worktree 中使用

---

## 📝 文件命名约定

- `*_env.py`: 环境相关
- `*_policy.py`: 策略相关
- `record_*.py`: 数据采集
- `imitate_*.py`: 训练/评估
- `visualize_*.py`: 可视化
- `robust*.py`: 鲁棒性测试
- `run_*.py`: 实验运行脚本

---

## 🚀 快速开始

1. **安装环境**
   ```bash
   conda env create -f conda_env.yaml
   conda activate aloha
   ```

2. **生成数据**
   ```bash
   python record_sim_episodes.py --task_name sim_transfer_cube_scripted
   ```

3. **训练模型**
   ```bash
   python imitate_episodes.py \
       --task_name sim_transfer_cube_scripted \
       --ckpt_dir training_results/test \
       --policy_class ACT \
       --num_epochs 100
   ```

4. **评估模型**
   ```bash
   python imitate_episodes.py --eval --ckpt_dir training_results/test
   ```

---

## 📚 相关资源

- 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)
- DETR: [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)
- MuJoCo: [Multi-Joint dynamics with Contact](https://mujoco.org/)

---

最后更新: 2026-02-20
