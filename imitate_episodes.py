import torch
import numpy as np
import os
import pickle
import argparse
import json
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
def tqdm(iterable, *args, **kwargs): return iterable
from einops import rearrange

# Ensure DETR's top-level `util` package is importable without PYTHONPATH hacks.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DETR_PATH = os.path.join(_REPO_ROOT, 'detr')
if _DETR_PATH not in sys.path:
    sys.path.insert(0, _DETR_PATH)

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
from robustness import (
    apply_visual_perturbation,
    build_robust_scenarios,
    default_dynamics_suite_for_task,
    default_vision_suite_for_task,
    robust_output_filenames,
    sanitize_for_json,
    summarize_robust_results,
    SUPPORTED_DYNAMICS_SUITES,
    SUPPORTED_VISION_SUITES,
)

import IPython
e = IPython.embed



# ========= Mac M1 暴力适配补丁 (开始) =========
# 强行把所有的 .cuda() 劫持为 .to('cpu')
# 这样你就不用去代码里一行行找了！
# 如果你想尝试 M1 GPU 加速，可以把 'cpu' 改成 'mps' (但不保证所有算子都支持)
def output_to_device(obj):
    return obj.to('cpu') 

torch.Tensor.cuda = output_to_device
torch.nn.Module.cuda = output_to_device
# ========= Mac M1 暴力适配补丁 (结束) =========


TRAIN_AUG_PROFILES = {
    'none': {
        'global_prob': 1.0,
        'gaussian_noise_prob': 0.5,
        'gaussian_noise_sigma_min': 2.0,
        'gaussian_noise_sigma_max': 18.0,
        'motion_blur_prob': 0.25,
        'motion_blur_kernels': [3, 5],
        'brightness_contrast_prob': 0.4,
        'brightness_alpha_min': 0.85,
        'brightness_alpha_max': 1.15,
        'brightness_beta_min': -20.0,
        'brightness_beta_max': 20.0,
        'occlusion_prob': 0.25,
        'occlusion_area_min': 0.03,
        'occlusion_area_max': 0.12,
        'occlusion_fill_mode': 'black',
        'jpeg_prob': 0.2,
        'jpeg_quality_min': 35,
        'jpeg_quality_max': 85,
        'gamma_prob': 0.0,
        'gamma_min': 0.85,
        'gamma_max': 1.2,
        'color_shift_prob': 0.0,
        'color_hue_max': 0.02,
        'color_sat_scale_min': 0.85,
        'color_sat_scale_max': 1.15,
        'color_val_delta_min': -0.06,
        'color_val_delta_max': 0.06,
        'min_transforms': 0,
    },
    'vision_robust_v2': {
        'global_prob': 1.0,
        'gaussian_noise_prob': 0.62,
        'gaussian_noise_sigma_min': 3.0,
        'gaussian_noise_sigma_max': 24.0,
        'motion_blur_prob': 0.32,
        'motion_blur_kernels': [3, 5, 7],
        'brightness_contrast_prob': 0.45,
        'brightness_alpha_min': 0.8,
        'brightness_alpha_max': 1.2,
        'brightness_beta_min': -24.0,
        'brightness_beta_max': 24.0,
        'occlusion_prob': 0.35,
        'occlusion_area_min': 0.04,
        'occlusion_area_max': 0.18,
        'occlusion_fill_mode': 'mean',
        'jpeg_prob': 0.28,
        'jpeg_quality_min': 28,
        'jpeg_quality_max': 85,
        'gamma_prob': 0.18,
        'gamma_min': 0.78,
        'gamma_max': 1.35,
        'color_shift_prob': 0.18,
        'color_hue_max': 0.035,
        'color_sat_scale_min': 0.78,
        'color_sat_scale_max': 1.28,
        'color_val_delta_min': -0.1,
        'color_val_delta_max': 0.1,
        'min_transforms': 1,
    },
    'vision_robust_v3': {
        'global_prob': 1.0,
        'gaussian_noise_prob': 0.75,
        'gaussian_noise_sigma_min': 4.0,
        'gaussian_noise_sigma_max': 28.0,
        'motion_blur_prob': 0.35,
        'motion_blur_kernels': [3, 5, 7],
        'brightness_contrast_prob': 0.45,
        'brightness_alpha_min': 0.75,
        'brightness_alpha_max': 1.25,
        'brightness_beta_min': -28.0,
        'brightness_beta_max': 28.0,
        'occlusion_prob': 0.45,
        'occlusion_area_min': 0.04,
        'occlusion_area_max': 0.22,
        'occlusion_fill_mode': 'random',
        'jpeg_prob': 0.35,
        'jpeg_quality_min': 20,
        'jpeg_quality_max': 85,
        'gamma_prob': 0.3,
        'gamma_min': 0.72,
        'gamma_max': 1.5,
        'color_shift_prob': 0.3,
        'color_hue_max': 0.045,
        'color_sat_scale_min': 0.72,
        'color_sat_scale_max': 1.35,
        'color_val_delta_min': -0.14,
        'color_val_delta_max': 0.14,
        'min_transforms': 1,
    },
}

TRAIN_AUG_CURRICULUM_CHOICES = {'off', '2stage', '3stage'}
DEFAULT_STAGE_RATIOS = {
    '2stage': [0.4, 0.6],
    '3stage': [0.3, 0.4, 0.3],
}


def _copy_aug_config(cfg):
    copied = dict(cfg)
    copied['motion_blur_kernels'] = list(cfg.get('motion_blur_kernels', [3]))
    return copied


def _safe_scale_prob(value, scale):
    return float(np.clip(float(value) * float(scale), 0.0, 1.0))


def _mild_aug_config(base_cfg):
    cfg = _copy_aug_config(base_cfg)
    cfg['global_prob'] = _safe_scale_prob(cfg['global_prob'], 0.9)
    cfg['gaussian_noise_prob'] = _safe_scale_prob(cfg['gaussian_noise_prob'], 0.8)
    cfg['gaussian_noise_sigma_min'] = max(0.5, cfg['gaussian_noise_sigma_min'] * 0.75)
    cfg['gaussian_noise_sigma_max'] = max(cfg['gaussian_noise_sigma_min'], cfg['gaussian_noise_sigma_max'] * 0.75)
    cfg['motion_blur_prob'] = _safe_scale_prob(cfg['motion_blur_prob'], 0.85)
    cfg['brightness_contrast_prob'] = _safe_scale_prob(cfg['brightness_contrast_prob'], 0.85)
    cfg['occlusion_prob'] = _safe_scale_prob(cfg['occlusion_prob'], 0.75)
    cfg['occlusion_area_min'] = max(0.01, cfg['occlusion_area_min'] * 0.8)
    cfg['occlusion_area_max'] = max(cfg['occlusion_area_min'], cfg['occlusion_area_max'] * 0.8)
    cfg['jpeg_prob'] = _safe_scale_prob(cfg['jpeg_prob'], 0.85)
    cfg['gamma_prob'] = _safe_scale_prob(cfg['gamma_prob'], 0.8)
    cfg['color_shift_prob'] = _safe_scale_prob(cfg['color_shift_prob'], 0.8)
    cfg['min_transforms'] = max(0, int(cfg.get('min_transforms', 0)))
    return cfg


def _hard_aug_config(base_cfg):
    cfg = _copy_aug_config(base_cfg)
    cfg['global_prob'] = 1.0
    cfg['gaussian_noise_prob'] = _safe_scale_prob(cfg['gaussian_noise_prob'], 1.25)
    cfg['gaussian_noise_sigma_min'] = max(1.0, cfg['gaussian_noise_sigma_min'] * 1.1)
    cfg['gaussian_noise_sigma_max'] = min(45.0, max(cfg['gaussian_noise_sigma_min'], cfg['gaussian_noise_sigma_max'] * 1.4))
    cfg['motion_blur_prob'] = _safe_scale_prob(cfg['motion_blur_prob'], 1.2)
    cfg['motion_blur_kernels'] = sorted(set(list(cfg.get('motion_blur_kernels', [3])) + [7]))
    cfg['brightness_contrast_prob'] = _safe_scale_prob(cfg['brightness_contrast_prob'], 1.2)
    cfg['brightness_alpha_min'] = max(0.5, cfg['brightness_alpha_min'] - 0.08)
    cfg['brightness_alpha_max'] = min(1.6, cfg['brightness_alpha_max'] + 0.08)
    cfg['brightness_beta_min'] = max(-80.0, cfg['brightness_beta_min'] - 8.0)
    cfg['brightness_beta_max'] = min(80.0, cfg['brightness_beta_max'] + 8.0)
    cfg['occlusion_prob'] = _safe_scale_prob(cfg['occlusion_prob'], 1.25)
    cfg['occlusion_area_min'] = max(0.01, cfg['occlusion_area_min'] * 1.1)
    cfg['occlusion_area_max'] = min(0.4, max(cfg['occlusion_area_min'], cfg['occlusion_area_max'] * 1.5))
    cfg['jpeg_prob'] = _safe_scale_prob(cfg['jpeg_prob'], 1.2)
    cfg['jpeg_quality_min'] = max(10, int(cfg['jpeg_quality_min']) - 8)
    cfg['gamma_prob'] = _safe_scale_prob(cfg['gamma_prob'], 1.2)
    cfg['gamma_min'] = max(0.5, cfg['gamma_min'] - 0.1)
    cfg['gamma_max'] = min(2.0, cfg['gamma_max'] + 0.1)
    cfg['color_shift_prob'] = _safe_scale_prob(cfg['color_shift_prob'], 1.2)
    cfg['color_hue_max'] = min(0.15, cfg['color_hue_max'] + 0.015)
    cfg['color_sat_scale_min'] = max(0.4, cfg['color_sat_scale_min'] - 0.08)
    cfg['color_sat_scale_max'] = min(1.8, cfg['color_sat_scale_max'] + 0.08)
    cfg['color_val_delta_min'] = max(-0.4, cfg['color_val_delta_min'] - 0.03)
    cfg['color_val_delta_max'] = min(0.4, cfg['color_val_delta_max'] + 0.03)
    cfg['min_transforms'] = max(1, int(cfg.get('min_transforms', 1)))
    return cfg


def _apply_focus_flags(cfg, noise_focus=False, occlusion_focus=False):
    focused = _copy_aug_config(cfg)
    if noise_focus:
        focused['gaussian_noise_prob'] = min(1.0, focused['gaussian_noise_prob'] + 0.2)
        focused['gaussian_noise_sigma_max'] = min(50.0, focused['gaussian_noise_sigma_max'] + 6.0)
        focused['min_transforms'] = max(1, int(focused.get('min_transforms', 0)))
    if occlusion_focus:
        focused['occlusion_prob'] = min(1.0, focused['occlusion_prob'] + 0.15)
        focused['occlusion_area_max'] = min(0.45, focused['occlusion_area_max'] + 0.08)
        focused['occlusion_fill_mode'] = 'random'
        focused['min_transforms'] = max(1, int(focused.get('min_transforms', 0)))
    return focused


def _curriculum_stage_count(curriculum):
    if curriculum == '2stage':
        return 2
    if curriculum == '3stage':
        return 3
    return 1


def _parse_stage_ratios(stage_epochs_text, curriculum):
    stage_count = _curriculum_stage_count(curriculum)
    if stage_count == 1:
        return [1.0]

    if not stage_epochs_text:
        return list(DEFAULT_STAGE_RATIOS[curriculum])

    pieces = [p.strip() for p in stage_epochs_text.split(',') if p.strip()]
    if len(pieces) != stage_count:
        raise ValueError(f'train_aug_stage_epochs expects {stage_count} values for {curriculum}, got {len(pieces)}')
    try:
        raw_vals = [float(p) for p in pieces]
    except ValueError as exc:
        raise ValueError('train_aug_stage_epochs must be comma-separated floats') from exc
    if any(v <= 0.0 for v in raw_vals):
        raise ValueError('train_aug_stage_epochs values must be positive')
    total = float(sum(raw_vals))
    return [v / total for v in raw_vals]


def build_train_aug_config(args):
    profile_name = args.get('train_aug_profile', 'none')
    if profile_name not in TRAIN_AUG_PROFILES:
        raise ValueError(f'Unsupported train_aug_profile: {profile_name}')

    cfg = _copy_aug_config(TRAIN_AUG_PROFILES[profile_name])
    arg_to_cfg = {
        'train_aug_global_prob': 'global_prob',
        'train_aug_noise_prob': 'gaussian_noise_prob',
        'train_aug_noise_sigma_min': 'gaussian_noise_sigma_min',
        'train_aug_noise_sigma_max': 'gaussian_noise_sigma_max',
        'train_aug_blur_prob': 'motion_blur_prob',
        'train_aug_blur_kernels': 'motion_blur_kernels',
        'train_aug_brightness_prob': 'brightness_contrast_prob',
        'train_aug_brightness_alpha_min': 'brightness_alpha_min',
        'train_aug_brightness_alpha_max': 'brightness_alpha_max',
        'train_aug_brightness_beta_min': 'brightness_beta_min',
        'train_aug_brightness_beta_max': 'brightness_beta_max',
        'train_aug_occlusion_prob': 'occlusion_prob',
        'train_aug_occlusion_area_min': 'occlusion_area_min',
        'train_aug_occlusion_area_max': 'occlusion_area_max',
        'train_aug_occlusion_fill_mode': 'occlusion_fill_mode',
        'train_aug_jpeg_prob': 'jpeg_prob',
        'train_aug_jpeg_quality_min': 'jpeg_quality_min',
        'train_aug_jpeg_quality_max': 'jpeg_quality_max',
        'train_aug_gamma_prob': 'gamma_prob',
        'train_aug_gamma_min': 'gamma_min',
        'train_aug_gamma_max': 'gamma_max',
        'train_aug_color_shift_prob': 'color_shift_prob',
        'train_aug_color_hue_max': 'color_hue_max',
        'train_aug_color_sat_scale_min': 'color_sat_scale_min',
        'train_aug_color_sat_scale_max': 'color_sat_scale_max',
        'train_aug_color_val_delta_min': 'color_val_delta_min',
        'train_aug_color_val_delta_max': 'color_val_delta_max',
        'train_aug_min_transforms': 'min_transforms',
    }
    for arg_key, cfg_key in arg_to_cfg.items():
        value = args.get(arg_key)
        if value is not None:
            cfg[cfg_key] = value
    return cfg


def build_train_aug_schedule(base_aug_config, curriculum, stage_ratios, noise_focus, occlusion_focus):
    if curriculum == 'off':
        return [_copy_aug_config(base_aug_config)], [1.0]

    if curriculum == '2stage':
        stage_cfgs = [_mild_aug_config(base_aug_config), _hard_aug_config(base_aug_config)]
    elif curriculum == '3stage':
        stage_cfgs = [_mild_aug_config(base_aug_config), _copy_aug_config(base_aug_config), _hard_aug_config(base_aug_config)]
    else:
        raise ValueError(f'Unsupported train_aug_curriculum: {curriculum}')

    stage_cfgs[-1] = _apply_focus_flags(stage_cfgs[-1], noise_focus=noise_focus, occlusion_focus=occlusion_focus)
    return stage_cfgs, stage_ratios


def build_train_action_aug_config(args):
    cfg = {
        'action_noise_std': args.get('train_action_noise_std'),
        'action_delay_prob': args.get('train_action_delay_prob'),
        'action_delay_max_steps': args.get('train_action_delay_max_steps'),
        'action_aug_on_gripper': bool(args.get('train_action_aug_on_gripper', False)),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    if args.get('dataset_dir'):
        dataset_dir = args['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    if args.get('num_episodes') is not None:
        num_episodes = args['num_episodes']

    lr = args['lr'] if args.get('lr') is not None else 1e-5
    seed = args['seed'] if args.get('seed') is not None else 0
    num_epochs = args['num_epochs'] if args.get('num_epochs') is not None else 0
    robust_mode = args.get('robust_mode', 'none')
    robust_suite = args.get('robust_suite', 'none')
    robust_eval = bool(args.get('robust_eval', False))
    if robust_eval and robust_mode == 'vision_only' and robust_suite == 'none':
        robust_suite = default_vision_suite_for_task(task_name)
    if robust_eval and robust_mode == 'vision_dynamics' and robust_suite == 'none':
        robust_suite = default_vision_suite_for_task(task_name)
    if robust_eval and robust_mode == 'dynamics_only' and robust_suite == 'none':
        robust_suite = default_dynamics_suite_for_task(task_name)

    train_visual_aug = bool(args.get('train_visual_aug', False))
    train_aug_profile = args.get('train_aug_profile', 'none')
    train_aug_curriculum = args.get('train_aug_curriculum', 'off')
    train_aug_noise_focus = bool(args.get('train_aug_noise_focus', False))
    train_aug_occlusion_focus = bool(args.get('train_aug_occlusion_focus', False))
    num_workers = int(args.get('num_workers', 0))
    train_aug_stage_epochs = args.get('train_aug_stage_epochs', '')
    train_aug_config = build_train_aug_config(args) if train_visual_aug else None
    train_aug_stage_configs = None
    train_aug_stage_ratios = [1.0]
    if train_visual_aug:
        train_aug_stage_ratios = _parse_stage_ratios(train_aug_stage_epochs, train_aug_curriculum)
        train_aug_stage_configs, train_aug_stage_ratios = build_train_aug_schedule(
            train_aug_config,
            curriculum=train_aug_curriculum,
            stage_ratios=train_aug_stage_ratios,
            noise_focus=train_aug_noise_focus,
            occlusion_focus=train_aug_occlusion_focus,
        )
    train_action_aug = bool(args.get('train_action_aug', False))
    train_action_aug_config = build_train_action_aug_config(args) if train_action_aug else None
    resume_ckpt = args.get('resume_ckpt')

    # fixed parameters
    state_dim = 14
    lr_backbone = args['lr_backbone'] if args.get('lr_backbone') is not None else 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': lr,
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': lr, 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': lr,
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'robust_eval': robust_eval,
        'robust_mode': robust_mode,
        'robust_suite': robust_suite,
        'robust_seed': args.get('robust_seed', 2026),
        'robust_num_rollouts': args.get('robust_num_rollouts', 50),
        'robust_save_json': args.get('robust_save_json', True),
        'train_visual_aug': train_visual_aug,
        'train_aug_profile': train_aug_profile,
        'train_aug_curriculum': train_aug_curriculum,
        'train_aug_stage_ratios': train_aug_stage_ratios,
        'train_aug_stage_configs': train_aug_stage_configs,
        'train_aug_noise_focus': train_aug_noise_focus,
        'train_aug_occlusion_focus': train_aug_occlusion_focus,
        'train_aug_config': train_aug_config,
        'train_action_aug': train_action_aug,
        'train_action_aug_config': train_action_aug_config,
        'resume_ckpt': resume_ckpt,
        'num_workers': num_workers,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        save_episode = bool(args.get('save_episode', True))
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=save_episode)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    if train_visual_aug:
        print('[train_aug] enabled')
        print(f'[train_aug] profile={train_aug_profile} curriculum={train_aug_curriculum} '
              f'noise_focus={train_aug_noise_focus} occlusion_focus={train_aug_occlusion_focus}')
        print(json.dumps(train_aug_config, indent=2))
        if train_aug_stage_configs is not None:
            for stage_idx, stage_cfg in enumerate(train_aug_stage_configs):
                print(f'[train_aug] stage_{stage_idx} ratio={train_aug_stage_ratios[stage_idx]:.3f}')
                print(json.dumps(stage_cfg, indent=2))
    if train_action_aug:
        print('[train_action_aug] enabled')
        print(json.dumps(train_action_aug_config, indent=2))

    initial_aug_config = train_aug_stage_configs[0] if train_aug_stage_configs else train_aug_config
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        train_aug=train_visual_aug,
        aug_config=initial_aug_config,
        train_action_aug=train_action_aug,
        action_aug_config=train_action_aug_config,
        num_workers=num_workers,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, visual_cfg=None, rng=None):
    curr_images = []
    for cam_name in camera_names:
        raw_image = ts.observation['images'][cam_name]
        if visual_cfg is not None:
            raw_image = apply_visual_perturbation(raw_image, visual_cfg, rng)
        curr_image = rearrange(raw_image, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    robust_eval = bool(config.get('robust_eval', False))
    robust_mode = config.get('robust_mode', 'none')
    robust_suite = config.get('robust_suite', 'none')
    robust_seed = int(config.get('robust_seed', 2026))
    num_rollouts = int(config.get('robust_num_rollouts', 50))
    robust_save_json = bool(config.get('robust_save_json', True))

    if robust_mode not in {'none', 'vision_only', 'dynamics_only', 'vision_dynamics'}:
        raise ValueError(f'Unsupported robust_mode: {robust_mode}')
    supports_dynamics_task = ('sim_insertion' in task_name) or ('sim_transfer_cube' in task_name)
    if robust_eval and robust_mode in {'dynamics_only', 'vision_dynamics'} and (real_robot or not supports_dynamics_task):
        raise ValueError('dynamics_only/vision_dynamics robust eval currently supports simulated transfer_cube/insertion tasks only')
    if robust_eval and robust_mode == 'vision_only' and robust_suite == 'none':
        robust_suite = default_vision_suite_for_task(task_name)
    if robust_eval and robust_mode == 'vision_dynamics' and robust_suite == 'none':
        robust_suite = default_vision_suite_for_task(task_name)
    if robust_eval and robust_mode == 'dynamics_only' and robust_suite == 'none':
        robust_suite = default_dynamics_suite_for_task(task_name)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment factory
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = None
        env_max_reward = None

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    if robust_eval and robust_mode != 'none':
        scenarios = build_robust_scenarios(robust_mode, robust_suite, robust_seed, num_rollouts)
    else:
        scenarios = build_robust_scenarios('none', 'none', robust_seed, num_rollouts)

    episode_returns = []
    highest_rewards = []
    robust_rollout_results = []
    for rollout_id in range(num_rollouts):
        scenario = scenarios[rollout_id]
        rollout_rng = np.random.default_rng(int(scenario['seed']))

        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        if not real_robot:
            dynamics_cfg = scenario['dynamics_cfg'] if robust_eval and robust_mode in {'dynamics_only', 'vision_dynamics'} else None
            env = make_sim_env(task_name, dynamics_cfg=dynamics_cfg)
            env_max_reward = env.task.max_reward

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                visual_cfg = scenario['visual_cfg'] if robust_eval and robust_mode in {'vision_only', 'vision_dynamics'} else None
                curr_image = get_image(ts, camera_names, visual_cfg=visual_cfg, rng=rollout_rng)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = float(np.sum(rewards[rewards != None]))
        episode_returns.append(episode_return)
        episode_highest_reward = float(np.max(rewards))
        highest_rewards.append(episode_highest_reward)
        rollout_success = bool(episode_highest_reward == env_max_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {rollout_success}')

        robust_rollout_results.append({
            'rollout_id': int(rollout_id),
            'scenario_id': scenario['scenario_id'],
            'mode': scenario['mode'],
            'suite': scenario['suite'],
            'visual_cfg': scenario['visual_cfg'],
            'dynamics_cfg': scenario['dynamics_cfg'],
            'episode_return': episode_return,
            'episode_highest_reward': episode_highest_reward,
            'env_max_reward': float(env_max_reward),
            'success': rollout_success,
        })

        if save_episode:
            if robust_eval and robust_mode in {'vision_only', 'dynamics_only', 'vision_dynamics'}:
                video_name = f'video_{robust_mode}_{rollout_id}.mp4'
            else:
                video_name = f'video{rollout_id}.mp4'
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, video_name))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    if robust_eval and robust_mode in {'vision_only', 'dynamics_only', 'vision_dynamics'}:
        result_file_name = f"result_{ckpt_name.split('.')[0]}_{robust_mode}.txt"
    else:
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    if robust_eval and robust_mode in {'vision_only', 'dynamics_only', 'vision_dynamics'} and robust_save_json:
        results_name, summary_name = robust_output_filenames(robust_mode)
        results_path = os.path.join(ckpt_dir, results_name)
        summary_path = os.path.join(ckpt_dir, summary_name)
        with open(results_path, 'w') as f:
            json.dump(sanitize_for_json(robust_rollout_results), f, indent=2)
        with open(summary_path, 'w') as f:
            summary = summarize_robust_results(robust_rollout_results, robust_mode, robust_suite)
            json.dump(sanitize_for_json(summary), f, indent=2)
        print(f'Saved robust results to {results_path}')
        print(f'Saved robust summary to {summary_path}')

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    resume_ckpt = config.get('resume_ckpt')
    train_aug_stage_configs = config.get('train_aug_stage_configs') or []
    train_aug_stage_ratios = config.get('train_aug_stage_ratios') or [1.0]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if resume_ckpt is not None:
        if not os.path.isabs(resume_ckpt):
            resume_ckpt = os.path.join(ckpt_dir, resume_ckpt)
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f'resume checkpoint not found: {resume_ckpt}')
        print(f'Loading resume checkpoint: {resume_ckpt}')
        loading_status = policy.load_state_dict(torch.load(resume_ckpt, map_location='cpu'))
        print(loading_status)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    active_aug_stage = None
    train_dataset = getattr(train_dataloader, 'dataset', None)
    if train_aug_stage_configs:
        if len(train_aug_stage_configs) != len(train_aug_stage_ratios):
            raise ValueError('train_aug_stage_configs and train_aug_stage_ratios must have the same length')
        stage_ratio_sum = float(sum(float(r) for r in train_aug_stage_ratios))
        if stage_ratio_sum <= 0:
            raise ValueError('train_aug_stage_ratios sum must be positive')
        train_aug_stage_ratios = [float(r) / stage_ratio_sum for r in train_aug_stage_ratios]
    for epoch in tqdm(range(num_epochs)):
        if train_aug_stage_configs and train_dataset is not None:
            progress = (epoch + 0.5) / max(1, num_epochs)
            cumulative = 0.0
            stage_idx = len(train_aug_stage_configs) - 1
            for idx, ratio in enumerate(train_aug_stage_ratios):
                cumulative += ratio
                if progress <= cumulative or idx == len(train_aug_stage_ratios) - 1:
                    stage_idx = idx
                    break
            if stage_idx != active_aug_stage:
                train_dataset.set_aug_config(train_aug_stage_configs[stage_idx])
                train_dataset.set_train_aug(True)
                active_aug_stage = stage_idx
                print(f'[train_aug] switched to curriculum stage {stage_idx} at epoch {epoch}')

        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--num_workers', action='store', type=int, default=0,
                        help='DataLoader worker count (set 0 for macOS OMP stability)')
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=None)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False, default=None)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=False, default=None)
    parser.add_argument('--lr_backbone', action='store', type=float, help='backbone lr', required=False, default=None)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='resume/fine-tune checkpoint path', required=False, default=None)
    parser.add_argument('--dataset_dir', action='store', type=str, help='override dataset directory', required=False, default=None)
    parser.add_argument('--num_episodes', action='store', type=int, help='override number of episodes', required=False, default=None)
    parser.add_argument('--train_visual_aug', action='store_true', help='enable train-time visual augmentation')
    parser.add_argument('--train_aug_profile', action='store', type=str, default='none',
                        choices=sorted(TRAIN_AUG_PROFILES.keys()))
    parser.add_argument('--train_aug_curriculum', action='store', type=str, default='off',
                        choices=sorted(TRAIN_AUG_CURRICULUM_CHOICES))
    parser.add_argument('--train_aug_stage_epochs', action='store', type=str, default='',
                        help='comma-separated stage ratios, e.g. 0.3,0.4,0.3 for 3stage')
    parser.add_argument('--train_aug_noise_focus', action='store_true')
    parser.add_argument('--train_aug_occlusion_focus', action='store_true')
    parser.add_argument('--train_aug_global_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_noise_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_noise_sigma_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_noise_sigma_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_blur_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_blur_kernels', action='store', nargs='+', type=int, default=None)
    parser.add_argument('--train_aug_brightness_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_brightness_alpha_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_brightness_alpha_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_brightness_beta_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_brightness_beta_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_occlusion_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_occlusion_area_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_occlusion_area_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_occlusion_fill_mode', action='store', type=str, default=None,
                        choices=['black', 'mean', 'random'])
    parser.add_argument('--train_aug_jpeg_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_jpeg_quality_min', action='store', type=int, default=None)
    parser.add_argument('--train_aug_jpeg_quality_max', action='store', type=int, default=None)
    parser.add_argument('--train_aug_gamma_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_gamma_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_gamma_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_shift_prob', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_hue_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_sat_scale_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_sat_scale_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_val_delta_min', action='store', type=float, default=None)
    parser.add_argument('--train_aug_color_val_delta_max', action='store', type=float, default=None)
    parser.add_argument('--train_aug_min_transforms', action='store', type=int, default=None)
    parser.add_argument('--train_action_aug', action='store_true', help='enable train-time action augmentation')
    parser.add_argument('--train_action_noise_std', action='store', type=float, default=None,
                        help='std for Gaussian noise in normalized action space')
    parser.add_argument('--train_action_delay_prob', action='store', type=float, default=None,
                        help='probability of random action delay per timestep')
    parser.add_argument('--train_action_delay_max_steps', action='store', type=int, default=None,
                        help='maximum delayed steps when action delay is triggered')
    parser.add_argument('--train_action_aug_on_gripper', action='store_true',
                        help='also augment gripper action dims')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--robust_eval', action='store_true')
    parser.add_argument('--robust_mode', action='store', type=str, default='none',
                        choices=['none', 'vision_only', 'dynamics_only', 'vision_dynamics'])
    parser.add_argument('--robust_suite', action='store', type=str, default='none')
    parser.add_argument('--robust_seed', action='store', type=int, default=2026)
    parser.add_argument('--robust_num_rollouts', action='store', type=int, default=50)
    parser.add_argument('--robust_save_json', action='store_true', dest='robust_save_json')
    parser.add_argument('--no_robust_save_json', action='store_false', dest='robust_save_json')
    parser.add_argument('--save_episode', action='store_true', dest='save_episode')
    parser.add_argument('--no_save_episode', action='store_false', dest='save_episode')
    parser.set_defaults(robust_save_json=True)
    parser.set_defaults(save_episode=True)
    
    parsed_args = parser.parse_args()
    if not parsed_args.eval:
        missing = []
        if parsed_args.seed is None:
            missing.append('--seed')
        if parsed_args.num_epochs is None:
            missing.append('--num_epochs')
        if parsed_args.lr is None:
            missing.append('--lr')
        if missing:
            parser.error(f"the following arguments are required when not using --eval: {', '.join(missing)}")
    if parsed_args.robust_eval and parsed_args.robust_mode == 'none':
        parser.error('--robust_eval requires --robust_mode to be vision_only, dynamics_only, or vision_dynamics')
    if parsed_args.robust_mode in {'vision_only', 'vision_dynamics'} and parsed_args.robust_suite not in ({'none'} | SUPPORTED_VISION_SUITES):
        parser.error(f'{parsed_args.robust_mode} supports --robust_suite in {{"none", *SUPPORTED_VISION_SUITES}}')
    if parsed_args.robust_mode == 'dynamics_only' and parsed_args.robust_suite not in ({'none'} | SUPPORTED_DYNAMICS_SUITES):
        parser.error(f'dynamics_only supports --robust_suite in {{"none", *SUPPORTED_DYNAMICS_SUITES}}')
    if parsed_args.robust_mode in {'dynamics_only', 'vision_dynamics'} and parsed_args.task_name[:4] == 'sim_' and ('sim_insertion' not in parsed_args.task_name and 'sim_transfer_cube' not in parsed_args.task_name):
        parser.error('dynamics_only/vision_dynamics currently supports sim_transfer_cube* and sim_insertion*')
    if parsed_args.robust_num_rollouts <= 0:
        parser.error('--robust_num_rollouts must be positive')
    if parsed_args.num_workers < 0:
        parser.error('--num_workers must be >= 0')
    if parsed_args.train_aug_curriculum != 'off' and not parsed_args.train_visual_aug:
        parser.error('--train_aug_curriculum requires --train_visual_aug')
    if parsed_args.train_aug_stage_epochs and parsed_args.train_aug_curriculum == 'off':
        parser.error('--train_aug_stage_epochs requires --train_aug_curriculum 2stage or 3stage')
    try:
        _parse_stage_ratios(parsed_args.train_aug_stage_epochs, parsed_args.train_aug_curriculum)
    except ValueError as exc:
        parser.error(str(exc))

    aug_prob_fields = [
        'train_aug_global_prob',
        'train_aug_noise_prob',
        'train_aug_blur_prob',
        'train_aug_brightness_prob',
        'train_aug_occlusion_prob',
        'train_aug_jpeg_prob',
        'train_aug_gamma_prob',
        'train_aug_color_shift_prob',
    ]
    for field_name in aug_prob_fields:
        field_value = getattr(parsed_args, field_name)
        if field_value is None:
            continue
        if field_value < 0.0 or field_value > 1.0:
            parser.error(f'--{field_name} must be in [0, 1]')

    if parsed_args.train_aug_noise_sigma_min is not None and parsed_args.train_aug_noise_sigma_min < 0.0:
        parser.error('--train_aug_noise_sigma_min must be non-negative')
    if parsed_args.train_aug_noise_sigma_max is not None and parsed_args.train_aug_noise_sigma_max < 0.0:
        parser.error('--train_aug_noise_sigma_max must be non-negative')
    if (parsed_args.train_aug_noise_sigma_min is not None and
            parsed_args.train_aug_noise_sigma_max is not None and
            parsed_args.train_aug_noise_sigma_min > parsed_args.train_aug_noise_sigma_max):
        parser.error('--train_aug_noise_sigma_min must be <= --train_aug_noise_sigma_max')

    if parsed_args.train_aug_brightness_alpha_min is not None and parsed_args.train_aug_brightness_alpha_min <= 0.0:
        parser.error('--train_aug_brightness_alpha_min must be positive')
    if parsed_args.train_aug_brightness_alpha_max is not None and parsed_args.train_aug_brightness_alpha_max <= 0.0:
        parser.error('--train_aug_brightness_alpha_max must be positive')
    if (parsed_args.train_aug_brightness_alpha_min is not None and
            parsed_args.train_aug_brightness_alpha_max is not None and
            parsed_args.train_aug_brightness_alpha_min > parsed_args.train_aug_brightness_alpha_max):
        parser.error('--train_aug_brightness_alpha_min must be <= --train_aug_brightness_alpha_max')

    if (parsed_args.train_aug_brightness_beta_min is not None and
            parsed_args.train_aug_brightness_beta_max is not None and
            parsed_args.train_aug_brightness_beta_min > parsed_args.train_aug_brightness_beta_max):
        parser.error('--train_aug_brightness_beta_min must be <= --train_aug_brightness_beta_max')

    if parsed_args.train_aug_occlusion_area_min is not None and (
            parsed_args.train_aug_occlusion_area_min < 0.0 or parsed_args.train_aug_occlusion_area_min > 1.0):
        parser.error('--train_aug_occlusion_area_min must be in [0, 1]')
    if parsed_args.train_aug_occlusion_area_max is not None and (
            parsed_args.train_aug_occlusion_area_max < 0.0 or parsed_args.train_aug_occlusion_area_max > 1.0):
        parser.error('--train_aug_occlusion_area_max must be in [0, 1]')
    if (parsed_args.train_aug_occlusion_area_min is not None and
            parsed_args.train_aug_occlusion_area_max is not None and
            parsed_args.train_aug_occlusion_area_min > parsed_args.train_aug_occlusion_area_max):
        parser.error('--train_aug_occlusion_area_min must be <= --train_aug_occlusion_area_max')

    if parsed_args.train_aug_jpeg_quality_min is not None and (
            parsed_args.train_aug_jpeg_quality_min < 1 or parsed_args.train_aug_jpeg_quality_min > 100):
        parser.error('--train_aug_jpeg_quality_min must be in [1, 100]')
    if parsed_args.train_aug_jpeg_quality_max is not None and (
            parsed_args.train_aug_jpeg_quality_max < 1 or parsed_args.train_aug_jpeg_quality_max > 100):
        parser.error('--train_aug_jpeg_quality_max must be in [1, 100]')
    if (parsed_args.train_aug_jpeg_quality_min is not None and
            parsed_args.train_aug_jpeg_quality_max is not None and
            parsed_args.train_aug_jpeg_quality_min > parsed_args.train_aug_jpeg_quality_max):
        parser.error('--train_aug_jpeg_quality_min must be <= --train_aug_jpeg_quality_max')

    if parsed_args.train_aug_gamma_min is not None and parsed_args.train_aug_gamma_min <= 0.0:
        parser.error('--train_aug_gamma_min must be positive')
    if parsed_args.train_aug_gamma_max is not None and parsed_args.train_aug_gamma_max <= 0.0:
        parser.error('--train_aug_gamma_max must be positive')
    if (parsed_args.train_aug_gamma_min is not None and
            parsed_args.train_aug_gamma_max is not None and
            parsed_args.train_aug_gamma_min > parsed_args.train_aug_gamma_max):
        parser.error('--train_aug_gamma_min must be <= --train_aug_gamma_max')

    if parsed_args.train_aug_color_hue_max is not None and (
            parsed_args.train_aug_color_hue_max < 0.0 or parsed_args.train_aug_color_hue_max > 0.5):
        parser.error('--train_aug_color_hue_max must be in [0, 0.5]')
    if parsed_args.train_aug_color_sat_scale_min is not None and parsed_args.train_aug_color_sat_scale_min <= 0.0:
        parser.error('--train_aug_color_sat_scale_min must be positive')
    if parsed_args.train_aug_color_sat_scale_max is not None and parsed_args.train_aug_color_sat_scale_max <= 0.0:
        parser.error('--train_aug_color_sat_scale_max must be positive')
    if (parsed_args.train_aug_color_sat_scale_min is not None and
            parsed_args.train_aug_color_sat_scale_max is not None and
            parsed_args.train_aug_color_sat_scale_min > parsed_args.train_aug_color_sat_scale_max):
        parser.error('--train_aug_color_sat_scale_min must be <= --train_aug_color_sat_scale_max')
    if parsed_args.train_aug_color_val_delta_min is not None and (
            parsed_args.train_aug_color_val_delta_min < -1.0 or parsed_args.train_aug_color_val_delta_min > 1.0):
        parser.error('--train_aug_color_val_delta_min must be in [-1, 1]')
    if parsed_args.train_aug_color_val_delta_max is not None and (
            parsed_args.train_aug_color_val_delta_max < -1.0 or parsed_args.train_aug_color_val_delta_max > 1.0):
        parser.error('--train_aug_color_val_delta_max must be in [-1, 1]')
    if (parsed_args.train_aug_color_val_delta_min is not None and
            parsed_args.train_aug_color_val_delta_max is not None and
            parsed_args.train_aug_color_val_delta_min > parsed_args.train_aug_color_val_delta_max):
        parser.error('--train_aug_color_val_delta_min must be <= --train_aug_color_val_delta_max')

    if parsed_args.train_aug_min_transforms is not None and parsed_args.train_aug_min_transforms < 0:
        parser.error('--train_aug_min_transforms must be non-negative')
    if parsed_args.train_aug_blur_kernels is not None and any(
            (kernel <= 0 or kernel % 2 == 0) for kernel in parsed_args.train_aug_blur_kernels):
        parser.error('--train_aug_blur_kernels only supports positive odd integers')
    if (parsed_args.train_action_noise_std is not None or
            parsed_args.train_action_delay_prob is not None or
            parsed_args.train_action_delay_max_steps is not None or
            parsed_args.train_action_aug_on_gripper) and not parsed_args.train_action_aug:
        parser.error('--train_action_* requires --train_action_aug')
    if parsed_args.train_action_noise_std is not None and parsed_args.train_action_noise_std < 0.0:
        parser.error('--train_action_noise_std must be non-negative')
    if parsed_args.train_action_delay_prob is not None and (
            parsed_args.train_action_delay_prob < 0.0 or parsed_args.train_action_delay_prob > 1.0):
        parser.error('--train_action_delay_prob must be in [0, 1]')
    if parsed_args.train_action_delay_max_steps is not None and parsed_args.train_action_delay_max_steps < 0:
        parser.error('--train_action_delay_max_steps must be non-negative')

    main(vars(parsed_args))
