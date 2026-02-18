import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

from robustness import apply_visual_perturbation

import IPython
e = IPython.embed

def _default_train_aug_config():
    return {
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
    }


def _build_train_aug_config(aug_config):
    cfg = _default_train_aug_config()
    if aug_config is not None:
        cfg.update({k: v for k, v in aug_config.items() if v is not None})

    prob_keys = [
        'global_prob',
        'gaussian_noise_prob',
        'motion_blur_prob',
        'brightness_contrast_prob',
        'occlusion_prob',
        'jpeg_prob',
        'gamma_prob',
        'color_shift_prob',
    ]
    for key in prob_keys:
        cfg[key] = float(np.clip(float(cfg[key]), 0.0, 1.0))

    cfg['gaussian_noise_sigma_min'] = float(max(0.0, cfg['gaussian_noise_sigma_min']))
    cfg['gaussian_noise_sigma_max'] = float(max(0.0, cfg['gaussian_noise_sigma_max']))
    if cfg['gaussian_noise_sigma_min'] > cfg['gaussian_noise_sigma_max']:
        cfg['gaussian_noise_sigma_min'], cfg['gaussian_noise_sigma_max'] = (
            cfg['gaussian_noise_sigma_max'],
            cfg['gaussian_noise_sigma_min'],
        )

    cfg['brightness_alpha_min'] = float(max(0.1, cfg['brightness_alpha_min']))
    cfg['brightness_alpha_max'] = float(max(0.1, cfg['brightness_alpha_max']))
    if cfg['brightness_alpha_min'] > cfg['brightness_alpha_max']:
        cfg['brightness_alpha_min'], cfg['brightness_alpha_max'] = (
            cfg['brightness_alpha_max'],
            cfg['brightness_alpha_min'],
        )

    cfg['brightness_beta_min'] = float(cfg['brightness_beta_min'])
    cfg['brightness_beta_max'] = float(cfg['brightness_beta_max'])
    if cfg['brightness_beta_min'] > cfg['brightness_beta_max']:
        cfg['brightness_beta_min'], cfg['brightness_beta_max'] = (
            cfg['brightness_beta_max'],
            cfg['brightness_beta_min'],
        )

    cfg['occlusion_area_min'] = float(np.clip(float(cfg['occlusion_area_min']), 0.0, 1.0))
    cfg['occlusion_area_max'] = float(np.clip(float(cfg['occlusion_area_max']), 0.0, 1.0))
    if cfg['occlusion_area_min'] > cfg['occlusion_area_max']:
        cfg['occlusion_area_min'], cfg['occlusion_area_max'] = (
            cfg['occlusion_area_max'],
            cfg['occlusion_area_min'],
        )
    cfg['occlusion_fill_mode'] = str(cfg.get('occlusion_fill_mode', 'black')).lower()
    if cfg['occlusion_fill_mode'] not in {'black', 'mean', 'random'}:
        cfg['occlusion_fill_mode'] = 'black'

    cfg['jpeg_quality_min'] = int(np.clip(int(cfg['jpeg_quality_min']), 1, 100))
    cfg['jpeg_quality_max'] = int(np.clip(int(cfg['jpeg_quality_max']), 1, 100))
    if cfg['jpeg_quality_min'] > cfg['jpeg_quality_max']:
        cfg['jpeg_quality_min'], cfg['jpeg_quality_max'] = (
            cfg['jpeg_quality_max'],
            cfg['jpeg_quality_min'],
        )

    kernels = []
    for kernel in cfg.get('motion_blur_kernels', []):
        kernel = int(kernel)
        if kernel > 0 and kernel % 2 == 1:
            kernels.append(kernel)
    cfg['motion_blur_kernels'] = kernels if kernels else [3]

    cfg['gamma_min'] = float(max(0.05, cfg.get('gamma_min', 0.85)))
    cfg['gamma_max'] = float(max(0.05, cfg.get('gamma_max', 1.2)))
    if cfg['gamma_min'] > cfg['gamma_max']:
        cfg['gamma_min'], cfg['gamma_max'] = cfg['gamma_max'], cfg['gamma_min']

    cfg['color_hue_max'] = float(np.clip(float(cfg.get('color_hue_max', 0.02)), 0.0, 0.5))
    cfg['color_sat_scale_min'] = float(max(0.05, cfg.get('color_sat_scale_min', 0.85)))
    cfg['color_sat_scale_max'] = float(max(0.05, cfg.get('color_sat_scale_max', 1.15)))
    if cfg['color_sat_scale_min'] > cfg['color_sat_scale_max']:
        cfg['color_sat_scale_min'], cfg['color_sat_scale_max'] = (
            cfg['color_sat_scale_max'],
            cfg['color_sat_scale_min'],
        )
    cfg['color_val_delta_min'] = float(np.clip(float(cfg.get('color_val_delta_min', -0.06)), -1.0, 1.0))
    cfg['color_val_delta_max'] = float(np.clip(float(cfg.get('color_val_delta_max', 0.06)), -1.0, 1.0))
    if cfg['color_val_delta_min'] > cfg['color_val_delta_max']:
        cfg['color_val_delta_min'], cfg['color_val_delta_max'] = (
            cfg['color_val_delta_max'],
            cfg['color_val_delta_min'],
        )

    cfg['min_transforms'] = int(max(0, int(cfg.get('min_transforms', 0))))

    return cfg


def _default_action_aug_config():
    return {
        'action_noise_std': 0.03,
        'action_delay_prob': 0.12,
        'action_delay_max_steps': 2,
        'action_aug_on_gripper': False,
    }


def _build_action_aug_config(action_aug_config):
    cfg = _default_action_aug_config()
    if action_aug_config is not None:
        cfg.update({k: v for k, v in action_aug_config.items() if v is not None})

    cfg['action_noise_std'] = float(max(0.0, cfg.get('action_noise_std', 0.03)))
    cfg['action_delay_prob'] = float(np.clip(float(cfg.get('action_delay_prob', 0.12)), 0.0, 1.0))
    cfg['action_delay_max_steps'] = int(max(0, int(cfg.get('action_delay_max_steps', 2))))
    cfg['action_aug_on_gripper'] = bool(cfg.get('action_aug_on_gripper', False))
    return cfg


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        is_train=False,
        train_aug=False,
        aug_config=None,
        train_action_aug=False,
        action_aug_config=None,
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_train = bool(is_train)
        self.train_aug = bool(train_aug)
        self.aug_config = _build_train_aug_config(aug_config)
        self.train_action_aug = bool(train_action_aug)
        self.action_aug_config = _build_action_aug_config(action_aug_config)
        self.rng = np.random.default_rng()
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def set_aug_config(self, aug_config):
        self.aug_config = _build_train_aug_config(aug_config)

    def set_train_aug(self, train_aug):
        self.train_aug = bool(train_aug)

    def set_action_aug_config(self, action_aug_config):
        self.action_aug_config = _build_action_aug_config(action_aug_config)

    def set_train_action_aug(self, train_action_aug):
        self.train_action_aug = bool(train_action_aug)

    def _sample_cfg_for_type(self, transform_type):
        if transform_type == 'brightness_contrast':
            return {
                'type': 'brightness_contrast',
                'alpha': float(self.rng.uniform(self.aug_config['brightness_alpha_min'], self.aug_config['brightness_alpha_max'])),
                'beta': float(self.rng.uniform(self.aug_config['brightness_beta_min'], self.aug_config['brightness_beta_max'])),
            }
        if transform_type == 'gaussian_noise':
            return {
                'type': 'gaussian_noise',
                'sigma': float(self.rng.uniform(self.aug_config['gaussian_noise_sigma_min'], self.aug_config['gaussian_noise_sigma_max'])),
            }
        if transform_type == 'occlusion':
            return {
                'type': 'occlusion',
                'area_ratio': float(self.rng.uniform(self.aug_config['occlusion_area_min'], self.aug_config['occlusion_area_max'])),
                'fill_mode': self.aug_config['occlusion_fill_mode'],
            }
        if transform_type == 'motion_blur':
            return {
                'type': 'motion_blur',
                'kernel_size': int(self.rng.choice(self.aug_config['motion_blur_kernels'])),
            }
        if transform_type == 'jpeg_compression':
            return {
                'type': 'jpeg_compression',
                'quality': int(self.rng.integers(self.aug_config['jpeg_quality_min'], self.aug_config['jpeg_quality_max'] + 1)),
            }
        if transform_type == 'gamma':
            return {
                'type': 'gamma',
                'gamma': float(self.rng.uniform(self.aug_config['gamma_min'], self.aug_config['gamma_max'])),
            }
        if transform_type == 'color_shift':
            hue_sign = -1.0 if self.rng.random() < 0.5 else 1.0
            return {
                'type': 'color_shift',
                'hue_delta': float(hue_sign * self.rng.uniform(0.0, self.aug_config['color_hue_max'])),
                'sat_scale': float(self.rng.uniform(self.aug_config['color_sat_scale_min'], self.aug_config['color_sat_scale_max'])),
                'val_delta': float(self.rng.uniform(self.aug_config['color_val_delta_min'], self.aug_config['color_val_delta_max'])),
            }
        raise ValueError(f'Unsupported train augment transform: {transform_type}')

    def _sample_visual_cfgs(self):
        if not (self.is_train and self.train_aug):
            return []
        if self.rng.random() > self.aug_config['global_prob']:
            return []

        transform_probs = [
            ('brightness_contrast', float(self.aug_config['brightness_contrast_prob'])),
            ('gaussian_noise', float(self.aug_config['gaussian_noise_prob'])),
            ('occlusion', float(self.aug_config['occlusion_prob'])),
            ('motion_blur', float(self.aug_config['motion_blur_prob'])),
            ('jpeg_compression', float(self.aug_config['jpeg_prob'])),
            ('gamma', float(self.aug_config['gamma_prob'])),
            ('color_shift', float(self.aug_config['color_shift_prob'])),
        ]
        transform_prob_map = dict(transform_probs)
        visual_cfgs = []
        selected_types = set()
        for transform_type, prob in transform_probs:
            if self.rng.random() < prob:
                visual_cfgs.append(self._sample_cfg_for_type(transform_type))
                selected_types.add(transform_type)

        min_required = int(self.aug_config.get('min_transforms', 0))
        if min_required > 0 and len(visual_cfgs) < min_required:
            remaining_types = [t for t, _ in transform_probs if t not in selected_types]
            while remaining_types and len(visual_cfgs) < min_required:
                weights = np.array(
                    [max(transform_prob_map[name], 1e-4) for name in remaining_types],
                    dtype=np.float64,
                )
                weights = weights / weights.sum()
                pick_idx = int(self.rng.choice(len(remaining_types), p=weights))
                picked_type = remaining_types.pop(pick_idx)
                visual_cfgs.append(self._sample_cfg_for_type(picked_type))
                selected_types.add(picked_type)

        if visual_cfgs:
            self.rng.shuffle(visual_cfgs)
        return visual_cfgs

    def _augment_image(self, image):
        visual_cfgs = self._sample_visual_cfgs()
        if not visual_cfgs:
            return image

        perturbed = image.copy()
        for visual_cfg in visual_cfgs:
            perturbed = apply_visual_perturbation(perturbed, visual_cfg, self.rng)
        return perturbed

    def _action_aug_indices(self, action_dim):
        if self.action_aug_config.get('action_aug_on_gripper', False):
            return list(range(action_dim))
        if action_dim >= 14:
            return [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        return list(range(action_dim))

    def _augment_action(self, action_data, action_len):
        if not (self.is_train and self.train_action_aug):
            return action_data
        if action_len <= 0:
            return action_data

        aug_action = action_data.clone()
        valid_action = aug_action[:action_len]
        action_dim = int(valid_action.shape[-1])
        target_indices = self._action_aug_indices(action_dim)
        if not target_indices:
            return aug_action

        noise_std = float(self.action_aug_config.get('action_noise_std', 0.0))
        if noise_std > 0.0:
            noise = torch.randn((action_len, len(target_indices)), dtype=valid_action.dtype) * noise_std
            valid_action[:, target_indices] = valid_action[:, target_indices] + noise

        delay_prob = float(self.action_aug_config.get('action_delay_prob', 0.0))
        delay_max_steps = int(self.action_aug_config.get('action_delay_max_steps', 0))
        if delay_prob > 0.0 and delay_max_steps > 0 and action_len > 1:
            for timestep in range(1, action_len):
                if self.rng.random() >= delay_prob:
                    continue
                delay = int(self.rng.integers(1, min(delay_max_steps, timestep) + 1))
                valid_action[timestep, target_indices] = valid_action[timestep - delay, target_indices]

        aug_action[:action_len] = valid_action
        return aug_action

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(self._augment_image(image_dict[cam_name]))
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = self._augment_action(action_data, int(action_len))
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    train_aug=False,
    aug_config=None,
    train_action_aug=False,
    action_aug_config=None,
    num_workers=0,
):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        is_train=True,
        train_aug=train_aug,
        aug_config=aug_config,
        train_action_aug=train_action_aug,
        action_aug_config=action_aug_config,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        is_train=False,
        train_aug=False,
        aug_config=None,
        train_action_aug=False,
        action_aug_config=None,
    )
    loader_kwargs = {
        'pin_memory': True,
        'num_workers': int(num_workers),
    }
    if int(num_workers) > 0:
        loader_kwargs['prefetch_factor'] = 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        **loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        **loader_kwargs,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
