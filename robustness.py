import io
from collections import defaultdict

import numpy as np


DEFAULT_DYNAMICS_SUITE_INSERTION = 'insertion_dynamics_v1'
DEFAULT_DYNAMICS_SUITE_TRANSFER = 'transfer_dynamics_v1'
DEFAULT_VISION_SUITE_INSERTION = 'insertion_vision_v1'
DEFAULT_VISION_SUITE_TRANSFER = 'transfer_vision_v1'
DEFAULT_VISION_SUITE_TRANSFER_V2 = 'transfer_vision_v2'

# Backward-compatible aliases
DEFAULT_DYNAMICS_SUITE = DEFAULT_DYNAMICS_SUITE_INSERTION
DEFAULT_VISION_SUITE = DEFAULT_VISION_SUITE_INSERTION

SUPPORTED_DYNAMICS_SUITES = {
    DEFAULT_DYNAMICS_SUITE_INSERTION,
    DEFAULT_DYNAMICS_SUITE_TRANSFER,
}
SUPPORTED_VISION_SUITES = {
    DEFAULT_VISION_SUITE_INSERTION,
    DEFAULT_VISION_SUITE_TRANSFER,
    DEFAULT_VISION_SUITE_TRANSFER_V2,
}


def default_dynamics_suite_for_task(task_name):
    if task_name and 'sim_transfer_cube' in task_name:
        return DEFAULT_DYNAMICS_SUITE_TRANSFER
    return DEFAULT_DYNAMICS_SUITE_INSERTION


def default_vision_suite_for_task(task_name):
    if task_name and 'sim_transfer_cube' in task_name:
        return DEFAULT_VISION_SUITE_TRANSFER
    return DEFAULT_VISION_SUITE_INSERTION


def _to_float(value):
    return float(np.asarray(value))


def _sanitize_dict(data):
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _sanitize_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_dict(v) for v in data]
    if isinstance(data, tuple):
        return [_sanitize_dict(v) for v in data]
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    return data


def _build_visual_cfg_transfer_v2(rng):
    perturbation_type = rng.choice(
        ['gaussian_noise', 'motion_blur', 'brightness_contrast', 'occlusion', 'jpeg_compression']
    )
    severity = str(rng.choice(['mild', 'moderate', 'severe']))

    if perturbation_type == 'gaussian_noise':
        sigma_ranges = {
            'mild': (4.0, 10.0),
            'moderate': (10.0, 20.0),
            'severe': (20.0, 35.0),
        }
        lo, hi = sigma_ranges[severity]
        return {
            'type': perturbation_type,
            'sigma': float(rng.uniform(lo, hi)),
            'severity': severity,
        }
    if perturbation_type == 'motion_blur':
        kernels = {
            'mild': [3],
            'moderate': [5, 7],
            'severe': [7, 9],
        }
        return {
            'type': perturbation_type,
            'kernel_size': int(rng.choice(kernels[severity])),
            'severity': severity,
        }
    if perturbation_type == 'brightness_contrast':
        ranges = {
            'mild': (0.88, 1.12, -15.0, 15.0),
            'moderate': (0.78, 1.22, -28.0, 28.0),
            'severe': (0.65, 1.35, -45.0, 45.0),
        }
        alpha_lo, alpha_hi, beta_lo, beta_hi = ranges[severity]
        return {
            'type': perturbation_type,
            'alpha': float(rng.uniform(alpha_lo, alpha_hi)),
            'beta': float(rng.uniform(beta_lo, beta_hi)),
            'severity': severity,
        }
    if perturbation_type == 'occlusion':
        area_ranges = {
            'mild': (0.03, 0.08),
            'moderate': (0.08, 0.16),
            'severe': (0.16, 0.28),
        }
        lo, hi = area_ranges[severity]
        fill_mode = str(rng.choice(['black', 'mean', 'random']))
        return {
            'type': perturbation_type,
            'area_ratio': float(rng.uniform(lo, hi)),
            'fill_mode': fill_mode,
            'severity': severity,
        }

    quality_ranges = {
        'mild': (60, 85),
        'moderate': (35, 65),
        'severe': (15, 40),
    }
    lo, hi = quality_ranges[severity]
    return {
        'type': perturbation_type,
        'quality': int(rng.integers(lo, hi + 1)),
        'severity': severity,
    }


def _build_visual_cfg(suite, rng):
    if suite not in SUPPORTED_VISION_SUITES:
        raise ValueError(f'Unsupported vision suite: {suite}')

    if suite == DEFAULT_VISION_SUITE_TRANSFER_V2:
        return _build_visual_cfg_transfer_v2(rng)

    perturbation_type = rng.choice(
        ['gaussian_noise', 'motion_blur', 'brightness_contrast', 'occlusion', 'jpeg_compression']
    )
    if perturbation_type == 'gaussian_noise':
        return {
            'type': perturbation_type,
            'sigma': float(rng.uniform(5.0, 30.0)),
        }
    if perturbation_type == 'motion_blur':
        return {
            'type': perturbation_type,
            'kernel_size': int(rng.choice([3, 5, 7])),
        }
    if perturbation_type == 'brightness_contrast':
        return {
            'type': perturbation_type,
            'alpha': float(rng.uniform(0.7, 1.3)),
            'beta': float(rng.uniform(-30.0, 30.0)),
        }
    if perturbation_type == 'occlusion':
        return {
            'type': perturbation_type,
            'area_ratio': float(rng.uniform(0.05, 0.20)),
            'fill_value': 0,
        }
    return {
        'type': perturbation_type,
        'quality': int(rng.integers(20, 81)),
    }


def _severity_bucket_from_scales(scales):
    max_delta = max(abs(float(scale) - 1.0) for scale in scales)
    if max_delta <= 0.10:
        return 'mild'
    if max_delta <= 0.20:
        return 'moderate'
    return 'severe'


def _build_dynamics_cfg(suite, rng):
    if suite not in SUPPORTED_DYNAMICS_SUITES:
        raise ValueError(f'Unsupported dynamics suite: {suite}')

    peg_mass_scale = float(rng.uniform(0.7, 1.3))
    socket_mass_scale = float(rng.uniform(0.7, 1.3))
    friction_scale = float(rng.uniform(0.5, 1.8))
    dof_damping_scale = float(rng.uniform(0.8, 1.2))
    actuator_gain_scale = float(rng.uniform(0.85, 1.15))
    scales = [peg_mass_scale, socket_mass_scale, friction_scale, dof_damping_scale, actuator_gain_scale]
    body_mass_scales = {'peg': peg_mass_scale, 'socket': socket_mass_scale}
    if suite == DEFAULT_DYNAMICS_SUITE_TRANSFER:
        body_mass_scales = {'box': peg_mass_scale}

    return {
        'type': 'dynamics_randomization',
        'body_mass_scales': body_mass_scales,
        'friction_scale': friction_scale,
        'dof_damping_scale': dof_damping_scale,
        'actuator_gain_scale': actuator_gain_scale,
        'severity': _severity_bucket_from_scales(scales),
    }


def _paired_dynamics_suite_for_vision_suite(vision_suite):
    if vision_suite == DEFAULT_VISION_SUITE_INSERTION:
        return DEFAULT_DYNAMICS_SUITE_INSERTION
    if vision_suite in {DEFAULT_VISION_SUITE_TRANSFER, DEFAULT_VISION_SUITE_TRANSFER_V2}:
        return DEFAULT_DYNAMICS_SUITE_TRANSFER
    raise ValueError(f'Unsupported vision suite for paired dynamics: {vision_suite}')


def build_robust_scenarios(mode, suite, seed, num_rollouts):
    if mode not in {'none', 'vision_only', 'dynamics_only', 'vision_dynamics'}:
        raise ValueError(f'Unsupported robust mode: {mode}')
    if num_rollouts <= 0:
        raise ValueError('num_rollouts must be positive')

    rng = np.random.default_rng(seed)
    scenarios = []
    for rollout_idx in range(num_rollouts):
        scenario_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        scenario_rng = np.random.default_rng(scenario_seed)
        scenario = {
            'scenario_id': f'{mode}_{rollout_idx:04d}',
            'mode': mode,
            'suite': suite,
            'seed': scenario_seed,
            'visual_cfg': None,
            'dynamics_cfg': None,
        }
        if mode == 'vision_only':
            scenario['visual_cfg'] = _build_visual_cfg(suite, scenario_rng)
        elif mode == 'dynamics_only':
            scenario['dynamics_cfg'] = _build_dynamics_cfg(suite, scenario_rng)
        elif mode == 'vision_dynamics':
            scenario['visual_cfg'] = _build_visual_cfg(suite, scenario_rng)
            dynamics_suite = _paired_dynamics_suite_for_vision_suite(suite)
            scenario['dynamics_cfg'] = _build_dynamics_cfg(dynamics_suite, scenario_rng)
        scenarios.append(scenario)
    return scenarios


def _apply_motion_blur(image, kernel_size):
    pad = kernel_size // 2
    image_f = image.astype(np.float32)
    padded = np.pad(image_f, ((0, 0), (pad, pad), (0, 0)), mode='edge')
    prefix = np.concatenate(
        [np.zeros((padded.shape[0], 1, padded.shape[2]), dtype=np.float32), np.cumsum(padded, axis=1, dtype=np.float32)],
        axis=1,
    )
    blurred = (prefix[:, kernel_size:] - prefix[:, :-kernel_size]) / float(kernel_size)
    return np.clip(blurred, 0.0, 255.0).astype(np.uint8)


def _apply_jpeg_compression(image, quality):
    try:
        import cv2

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format='JPEG', quality=int(quality))
        buffer.seek(0)
        return np.array(Image.open(buffer).convert('RGB'))
    except Exception:
        pass

    step = max(1, int(round((100 - quality) / 5)))
    compressed = (image // step) * step
    return compressed.astype(np.uint8)


def _apply_hsv_shift(image, hue_delta, sat_scale, val_delta):
    image_f = image.astype(np.float32) / 255.0
    try:
        import cv2

        hsv = cv2.cvtColor(image_f, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + (hue_delta * 360.0), 360.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2] + val_delta, 0.0, 1.0)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    except Exception:
        pass

    try:
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

        hsv = rgb_to_hsv(image_f)
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_delta, 1.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2] + val_delta, 0.0, 1.0)
        rgb = hsv_to_rgb(hsv)
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: keep semantics close even without HSV conversion utilities.
    scale = np.array([1.0 + hue_delta, sat_scale, 1.0 - hue_delta], dtype=np.float32).reshape(1, 1, 3)
    shifted = image.astype(np.float32) * scale + (val_delta * 255.0)
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def apply_visual_perturbation(image, cfg, rng):
    if cfg is None:
        return image
    if rng is None:
        rng = np.random.default_rng()

    perturbation_type = cfg.get('type')
    if perturbation_type == 'gaussian_noise':
        sigma = float(cfg['sigma'])
        noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
        image_f = image.astype(np.float32) + noise
        return np.clip(image_f, 0.0, 255.0).astype(np.uint8)

    if perturbation_type == 'motion_blur':
        kernel_size = int(cfg['kernel_size'])
        return _apply_motion_blur(image, kernel_size)

    if perturbation_type == 'brightness_contrast':
        alpha = float(cfg['alpha'])
        beta = float(cfg['beta'])
        image_f = image.astype(np.float32) * alpha + beta
        return np.clip(image_f, 0.0, 255.0).astype(np.uint8)

    if perturbation_type == 'gamma':
        gamma = float(cfg['gamma'])
        gamma = max(gamma, 1e-3)
        image_f = image.astype(np.float32) / 255.0
        corrected = np.power(np.clip(image_f, 0.0, 1.0), gamma)
        return np.clip(corrected * 255.0, 0.0, 255.0).astype(np.uint8)

    if perturbation_type == 'color_shift':
        hue_delta = float(cfg.get('hue_delta', 0.0))
        sat_scale = float(cfg.get('sat_scale', 1.0))
        val_delta = float(cfg.get('val_delta', 0.0))
        return _apply_hsv_shift(image, hue_delta=hue_delta, sat_scale=sat_scale, val_delta=val_delta)

    if perturbation_type == 'occlusion':
        h, w, _ = image.shape
        area_ratio = float(cfg['area_ratio'])
        fill_mode = str(cfg.get('fill_mode', '')).lower()
        fill_value = int(cfg.get('fill_value', 0))
        target_area = max(1, int(round(h * w * area_ratio)))
        aspect_ratio = float(rng.uniform(0.5, 2.0))
        occ_h = int(np.sqrt(target_area / aspect_ratio))
        occ_w = int(np.sqrt(target_area * aspect_ratio))
        occ_h = int(np.clip(occ_h, 1, h))
        occ_w = int(np.clip(occ_w, 1, w))
        top = int(rng.integers(0, h - occ_h + 1))
        left = int(rng.integers(0, w - occ_w + 1))
        perturbed = image.copy()
        if fill_mode == 'random':
            random_block = rng.integers(0, 256, size=(occ_h, occ_w, 3), dtype=np.uint8)
            perturbed[top:top + occ_h, left:left + occ_w, :] = random_block
        elif fill_mode == 'mean':
            mean_color = perturbed.reshape(-1, 3).mean(axis=0)
            perturbed[top:top + occ_h, left:left + occ_w, :] = mean_color.astype(np.uint8)
        else:
            perturbed[top:top + occ_h, left:left + occ_w, :] = fill_value
        return perturbed

    if perturbation_type == 'jpeg_compression':
        quality = int(cfg['quality'])
        return _apply_jpeg_compression(image, quality)

    raise ValueError(f'Unsupported visual perturbation type: {perturbation_type}')


def _group_stats(records):
    count = len(records)
    if count == 0:
        return {
            'count': 0,
            'success_rate': 0.0,
            'avg_return': 0.0,
            'avg_highest_reward': 0.0,
        }

    success_rate = sum(float(r['success']) for r in records) / count
    avg_return = sum(_to_float(r['episode_return']) for r in records) / count
    avg_highest_reward = sum(_to_float(r['episode_highest_reward']) for r in records) / count
    return {
        'count': int(count),
        'success_rate': float(success_rate),
        'avg_return': float(avg_return),
        'avg_highest_reward': float(avg_highest_reward),
    }


def _visual_severity_from_cfg(visual_cfg):
    if not visual_cfg:
        return None

    explicit = visual_cfg.get('severity')
    if explicit in {'mild', 'moderate', 'severe'}:
        return str(explicit)

    perturbation_type = visual_cfg.get('type')
    if perturbation_type == 'gaussian_noise':
        sigma = float(visual_cfg.get('sigma', 0.0))
        if sigma < 10.0:
            return 'mild'
        if sigma < 20.0:
            return 'moderate'
        return 'severe'
    if perturbation_type == 'motion_blur':
        kernel = int(visual_cfg.get('kernel_size', 3))
        if kernel <= 3:
            return 'mild'
        if kernel <= 7:
            return 'moderate'
        return 'severe'
    if perturbation_type == 'brightness_contrast':
        alpha = float(visual_cfg.get('alpha', 1.0))
        beta = float(visual_cfg.get('beta', 0.0))
        score = max(abs(alpha - 1.0) * 2.0, abs(beta) / 30.0)
        if score < 0.4:
            return 'mild'
        if score < 0.9:
            return 'moderate'
        return 'severe'
    if perturbation_type == 'occlusion':
        area_ratio = float(visual_cfg.get('area_ratio', 0.0))
        if area_ratio < 0.08:
            return 'mild'
        if area_ratio < 0.16:
            return 'moderate'
        return 'severe'
    if perturbation_type == 'jpeg_compression':
        quality = int(visual_cfg.get('quality', 100))
        if quality >= 60:
            return 'mild'
        if quality >= 35:
            return 'moderate'
        return 'severe'
    return 'unknown'


def summarize_robust_results(results, mode, suite):
    summary = {
        'mode': mode,
        'suite': suite,
        'overall': _group_stats(results),
        'by_visual_type': {},
        'by_visual_severity': {},
        'by_dynamics_severity': {},
    }

    by_visual_type = defaultdict(list)
    by_visual_severity = defaultdict(list)
    by_dynamics_severity = defaultdict(list)
    for row in results:
        visual_cfg = row.get('visual_cfg') or {}
        dynamics_cfg = row.get('dynamics_cfg') or {}
        if visual_cfg:
            by_visual_type[str(visual_cfg.get('type', 'unknown'))].append(row)
            visual_severity = _visual_severity_from_cfg(visual_cfg)
            if visual_severity:
                by_visual_severity[str(visual_severity)].append(row)
        if dynamics_cfg:
            severity = str(dynamics_cfg.get('severity', 'unknown'))
            by_dynamics_severity[severity].append(row)

    summary['by_visual_type'] = {k: _group_stats(v) for k, v in sorted(by_visual_type.items())}
    summary['by_visual_severity'] = {k: _group_stats(v) for k, v in sorted(by_visual_severity.items())}
    summary['by_dynamics_severity'] = {k: _group_stats(v) for k, v in sorted(by_dynamics_severity.items())}
    return _sanitize_dict(summary)


def robust_output_filenames(mode):
    if mode == 'dynamics_only':
        return 'robust_dynamics_results.json', 'robust_dynamics_summary.json'
    if mode == 'vision_only':
        return 'robust_vision_results.json', 'robust_vision_summary.json'
    if mode == 'vision_dynamics':
        return 'robust_vision_dynamics_results.json', 'robust_vision_dynamics_summary.json'
    return 'robust_results.json', 'robust_summary.json'


def sanitize_for_json(data):
    return _sanitize_dict(data)
