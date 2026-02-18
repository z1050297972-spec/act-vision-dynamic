#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from robustness import default_dynamics_suite_for_task, default_vision_suite_for_task

MODE_TO_SUMMARY_FILE = {
    'dynamics_only': 'robust_dynamics_summary.json',
    'vision_only': 'robust_vision_summary.json',
    'vision_dynamics': 'robust_vision_dynamics_summary.json',
}


def _run_command(cmd):
    print('[run] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _format_metric(value):
    return f'{float(value):.4f}'


def _build_text_report(dynamics_summary, vision_summary, vision_dynamics_summary, task_name, ckpt_dir):
    dyn_overall = dynamics_summary.get('overall', {})
    vis_overall = vision_summary.get('overall', {})
    combo_overall = (vision_dynamics_summary or {}).get('overall', {})

    dyn_success = float(dyn_overall.get('success_rate', 0.0))
    vis_success = float(vis_overall.get('success_rate', 0.0))
    dyn_return = float(dyn_overall.get('avg_return', 0.0))
    vis_return = float(vis_overall.get('avg_return', 0.0))
    combo_success = float(combo_overall.get('success_rate', 0.0))
    combo_return = float(combo_overall.get('avg_return', 0.0))

    harder_by_success = 'dynamics' if dyn_success < vis_success else 'vision'
    harder_by_return = 'dynamics' if dyn_return < vis_return else 'vision'

    lines = []
    lines.append('Robustness Comparison Report')
    lines.append(f'Timestamp: {datetime.now(timezone.utc).isoformat()}')
    lines.append(f'Task: {task_name}')
    lines.append(f'Checkpoint dir: {ckpt_dir}')
    lines.append('')
    lines.append('Overall')
    lines.append(f'- dynamics success_rate: {_format_metric(dyn_success)}')
    lines.append(f'- vision success_rate:   {_format_metric(vis_success)}')
    lines.append(f'- dynamics avg_return:   {_format_metric(dyn_return)}')
    lines.append(f'- vision avg_return:     {_format_metric(vis_return)}')
    if vision_dynamics_summary is not None:
        lines.append(f'- vision+dynamics success_rate: {_format_metric(combo_success)}')
        lines.append(f'- vision+dynamics avg_return:   {_format_metric(combo_return)}')
    lines.append(f'- success_rate_gap(dyn-vis): {_format_metric(dyn_success - vis_success)}')
    lines.append(f'- avg_return_gap(dyn-vis):   {_format_metric(dyn_return - vis_return)}')
    lines.append(f'- harder_by_success: {harder_by_success}')
    lines.append(f'- harder_by_return:  {harder_by_return}')
    lines.append('')

    lines.append('Dynamics Severity Breakdown')
    for key, value in sorted((dynamics_summary.get('by_dynamics_severity') or {}).items()):
        lines.append(
            f'- {key}: count={value.get("count", 0)} '
            f'success_rate={_format_metric(value.get("success_rate", 0.0))} '
            f'avg_return={_format_metric(value.get("avg_return", 0.0))}'
        )
    lines.append('')

    lines.append('Vision Type Breakdown')
    for key, value in sorted((vision_summary.get('by_visual_type') or {}).items()):
        lines.append(
            f'- {key}: count={value.get("count", 0)} '
            f'success_rate={_format_metric(value.get("success_rate", 0.0))} '
            f'avg_return={_format_metric(value.get("avg_return", 0.0))}'
        )
    lines.append('')
    if vision_dynamics_summary is not None:
        lines.append('Vision+Dynamics Severity Breakdown')
        for key, value in sorted((vision_dynamics_summary.get('by_dynamics_severity') or {}).items()):
            lines.append(
                f'- {key}: count={value.get("count", 0)} '
                f'success_rate={_format_metric(value.get("success_rate", 0.0))} '
                f'avg_return={_format_metric(value.get("avg_return", 0.0))}'
            )
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run dynamics-only, vision-only, and vision+dynamics robust evals, then produce a comparison report.'
    )
    parser.add_argument('--ckpt_dir', type=str, required=True, help='checkpoint directory')
    parser.add_argument('--task_name', type=str, default='sim_insertion_scripted')
    parser.add_argument('--policy_class', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--kl_weight', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--robust_seed', type=int, default=2026)
    parser.add_argument('--robust_num_rollouts', type=int, default=50)
    parser.add_argument('--python_bin', type=str, default=sys.executable)
    parser.add_argument('--eval_script', type=str, default='imitate_episodes.py')
    parser.add_argument('--dynamics_suite', type=str, default=None, help='optional override for dynamics suite')
    parser.add_argument('--vision_suite', type=str, default=None, help='optional override for vision suite')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--skip_runs', action='store_true', help='skip eval runs and only compare existing summaries')
    args, passthrough = parser.parse_known_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'ckpt_dir does not exist: {ckpt_dir}')
    if args.robust_num_rollouts <= 0:
        raise ValueError('robust_num_rollouts must be positive')
    if args.policy_class == 'ACT' and not args.skip_runs:
        missing = []
        if args.kl_weight is None:
            missing.append('--kl_weight')
        if args.chunk_size is None:
            missing.append('--chunk_size')
        if args.hidden_dim is None:
            missing.append('--hidden_dim')
        if args.dim_feedforward is None:
            missing.append('--dim_feedforward')
        if missing:
            raise ValueError(f'policy_class=ACT requires: {", ".join(missing)}')

    base_cmd = [
        args.python_bin,
        args.eval_script,
        '--eval',
        '--ckpt_dir',
        str(ckpt_dir),
        '--policy_class',
        args.policy_class,
        '--task_name',
        args.task_name,
        '--batch_size',
        str(args.batch_size),
        '--robust_eval',
        '--robust_seed',
        str(args.robust_seed),
        '--robust_num_rollouts',
        str(args.robust_num_rollouts),
        '--robust_save_json',
    ]
    if args.kl_weight is not None:
        base_cmd.extend(['--kl_weight', str(args.kl_weight)])
    if args.chunk_size is not None:
        base_cmd.extend(['--chunk_size', str(args.chunk_size)])
    if args.hidden_dim is not None:
        base_cmd.extend(['--hidden_dim', str(args.hidden_dim)])
    if args.dim_feedforward is not None:
        base_cmd.extend(['--dim_feedforward', str(args.dim_feedforward)])
    if args.onscreen_render:
        base_cmd.append('--onscreen_render')
    if args.temporal_agg:
        base_cmd.append('--temporal_agg')
    if passthrough:
        base_cmd.extend(passthrough)

    if not args.skip_runs:
        for mode in ['dynamics_only', 'vision_only', 'vision_dynamics']:
            if mode == 'dynamics_only':
                suite = args.dynamics_suite or default_dynamics_suite_for_task(args.task_name)
            else:
                suite = args.vision_suite or default_vision_suite_for_task(args.task_name)
            cmd = list(base_cmd) + ['--robust_mode', mode, '--robust_suite', suite]
            _run_command(cmd)

    dynamics_summary_path = ckpt_dir / MODE_TO_SUMMARY_FILE['dynamics_only']
    vision_summary_path = ckpt_dir / MODE_TO_SUMMARY_FILE['vision_only']
    vision_dynamics_summary_path = ckpt_dir / MODE_TO_SUMMARY_FILE['vision_dynamics']
    if not dynamics_summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {dynamics_summary_path}')
    if not vision_summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {vision_summary_path}')

    dynamics_summary = _read_json(dynamics_summary_path)
    vision_summary = _read_json(vision_summary_path)
    vision_dynamics_summary = None
    if vision_dynamics_summary_path.exists():
        vision_dynamics_summary = _read_json(vision_dynamics_summary_path)

    report_text = _build_text_report(
        dynamics_summary,
        vision_summary,
        vision_dynamics_summary,
        args.task_name,
        str(ckpt_dir),
    )
    report_path = ckpt_dir / 'robust_compare_report.txt'
    report_json_path = ckpt_dir / 'robust_compare_summary.json'

    comparison_payload = {
        'task_name': args.task_name,
        'ckpt_dir': str(ckpt_dir),
        'dynamics_summary_file': str(dynamics_summary_path),
        'vision_summary_file': str(vision_summary_path),
        'vision_dynamics_summary_file': str(vision_dynamics_summary_path) if vision_dynamics_summary is not None else None,
        'dynamics': dynamics_summary,
        'vision': vision_summary,
        'vision_dynamics': vision_dynamics_summary,
    }

    with open(report_path, 'w') as f:
        f.write(report_text + '\n')
    with open(report_json_path, 'w') as f:
        json.dump(comparison_payload, f, indent=2)

    print(report_text)
    print(f'Saved report: {report_path}')
    print(f'Saved json: {report_json_path}')


if __name__ == '__main__':
    main()
