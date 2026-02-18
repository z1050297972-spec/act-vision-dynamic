#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = {
    "A1": {
        "train_aug_profile": "vision_robust_v2",
        "train_aug_curriculum": "2stage",
        "train_aug_noise_focus": False,
        "train_aug_occlusion_focus": False,
    },
    "A2": {
        "train_aug_profile": "vision_robust_v3",
        "train_aug_curriculum": "2stage",
        "train_aug_noise_focus": False,
        "train_aug_occlusion_focus": False,
    },
    "A3": {
        "train_aug_profile": "vision_robust_v3",
        "train_aug_curriculum": "3stage",
        "train_aug_noise_focus": False,
        "train_aug_occlusion_focus": False,
    },
    "A4": {
        "train_aug_profile": "vision_robust_v3",
        "train_aug_curriculum": "3stage",
        "train_aug_noise_focus": True,
        "train_aug_occlusion_focus": False,
    },
    "A5": {
        "train_aug_profile": "vision_robust_v3",
        "train_aug_curriculum": "3stage",
        "train_aug_noise_focus": False,
        "train_aug_occlusion_focus": True,
    },
    "A6": {
        "train_aug_profile": "vision_robust_v3",
        "train_aug_curriculum": "3stage",
        "train_aug_noise_focus": True,
        "train_aug_occlusion_focus": True,
    },
}


IMITATE_EPISODES_BOOTSTRAP = "import runpy; runpy.run_path('imitate_episodes.py', run_name='__main__')"


MODE_SETTINGS = {
    "baseline": {
        "cli": [],
    },
    "vision": {
        "cli": ["--robust_eval", "--robust_mode", "vision_only", "--robust_suite", "transfer_vision_v2"],
    },
    "dynamics": {
        "cli": ["--robust_eval", "--robust_mode", "dynamics_only", "--robust_suite", "transfer_dynamics_v1"],
    },
}


def parse_int_list(text):
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    if not values:
        raise ValueError("list cannot be empty")
    return values


def parse_name_list(text):
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            values.append(piece)
    if not values:
        raise ValueError("name list cannot be empty")
    return values


def run_cmd(cmd, cwd, env, dry_run=False):
    cmd_str = " ".join(str(x) for x in cmd)
    print(f"[cmd] {cmd_str}", flush=True)
    if dry_run:
        return 0, ""

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def parse_eval_metrics(raw_text):
    success_matches = re.findall(r"Success rate:\s*([0-9]*\.?[0-9]+)", raw_text)
    return_matches = re.findall(r"Average return:\s*([-+]?[0-9]*\.?[0-9]+)", raw_text)
    if not success_matches or not return_matches:
        raise RuntimeError("failed to parse eval metrics from output")
    return float(success_matches[-1]), float(return_matches[-1])


def summarize_values(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    if len(values) == 1:
        return {
            "count": 1,
            "mean": float(values[0]),
            "std": 0.0,
            "min": float(values[0]),
            "max": float(values[0]),
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def maybe_add_common_eval_args(cmd, args, ckpt_dir, eval_seed):
    cmd.extend(
        [
            "--eval",
            "--task_name",
            args.task_name,
            "--ckpt_dir",
            str(ckpt_dir),
            "--policy_class",
            "ACT",
            "--batch_size",
            str(args.batch_size),
            "--kl_weight",
            str(args.kl_weight),
            "--chunk_size",
            str(args.chunk_size),
            "--hidden_dim",
            str(args.hidden_dim),
            "--dim_feedforward",
            str(args.dim_feedforward),
            "--robust_num_rollouts",
            str(args.eval_rollouts),
            "--robust_seed",
            str(eval_seed),
            "--no_save_episode",
            "--no_robust_save_json",
        ]
    )


def build_train_cmd(python_bin, args, ckpt_dir, train_seed, exp_cfg):
    cmd = [
        python_bin,
        "-c",
        IMITATE_EPISODES_BOOTSTRAP,
        "--task_name",
        args.task_name,
        "--ckpt_dir",
        str(ckpt_dir),
        "--policy_class",
        "ACT",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--kl_weight",
        str(args.kl_weight),
        "--chunk_size",
        str(args.chunk_size),
        "--hidden_dim",
        str(args.hidden_dim),
        "--dim_feedforward",
        str(args.dim_feedforward),
        "--num_epochs",
        str(args.num_epochs),
        "--lr",
        str(args.lr),
        "--lr_backbone",
        str(args.lr_backbone),
        "--seed",
        str(train_seed),
        "--train_visual_aug",
        "--train_aug_profile",
        exp_cfg["train_aug_profile"],
        "--train_aug_curriculum",
        exp_cfg["train_aug_curriculum"],
    ]
    if args.train_aug_stage_epochs:
        cmd.extend(["--train_aug_stage_epochs", args.train_aug_stage_epochs])
    if exp_cfg.get("train_aug_noise_focus", False):
        cmd.append("--train_aug_noise_focus")
    if exp_cfg.get("train_aug_occlusion_focus", False):
        cmd.append("--train_aug_occlusion_focus")
    if args.resume_ckpt:
        cmd.extend(["--resume_ckpt", args.resume_ckpt])
    if args.num_episodes is not None:
        cmd.extend(["--num_episodes", str(args.num_episodes)])
    if args.dataset_dir:
        cmd.extend(["--dataset_dir", args.dataset_dir])
    return cmd


def build_eval_cmd(python_bin, args, ckpt_dir, eval_seed, mode_name):
    cmd = [python_bin, "-c", IMITATE_EPISODES_BOOTSTRAP]
    maybe_add_common_eval_args(cmd, args, ckpt_dir, eval_seed)
    cmd.extend(MODE_SETTINGS[mode_name]["cli"])
    return cmd


def aggregate_records(records, gate_baseline, gate_vision):
    by_exp_mode = {}
    for row in records:
        key = (row["experiment"], row["mode"])
        by_exp_mode.setdefault(key, {"success": [], "return": []})
        by_exp_mode[key]["success"].append(row["success_rate"])
        by_exp_mode[key]["return"].append(row["avg_return"])

    exp_summary = {}
    for exp_name in sorted({r["experiment"] for r in records}):
        exp_summary[exp_name] = {}
        for mode_name in MODE_SETTINGS.keys():
            key = (exp_name, mode_name)
            mode_data = by_exp_mode.get(key, {"success": [], "return": []})
            exp_summary[exp_name][mode_name] = {
                "success_rate": summarize_values(mode_data["success"]),
                "avg_return": summarize_values(mode_data["return"]),
            }

        baseline_mean = exp_summary[exp_name]["baseline"]["success_rate"]["mean"]
        vision_mean = exp_summary[exp_name]["vision"]["success_rate"]["mean"]
        exp_summary[exp_name]["gate"] = {
            "baseline_min": gate_baseline,
            "vision_min": gate_vision,
            "pass_baseline": bool(baseline_mean is not None and baseline_mean >= gate_baseline),
            "pass_vision": bool(vision_mean is not None and vision_mean >= gate_vision),
            "pass_all": bool(
                baseline_mean is not None
                and vision_mean is not None
                and baseline_mean >= gate_baseline
                and vision_mean >= gate_vision
            ),
        }
    return exp_summary


def main():
    parser = argparse.ArgumentParser(description="Run A1~A6 visual-robustness experiment matrix.")
    parser.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output_root", type=str, default="training_results/vision_robust_matrix")
    parser.add_argument("--task_name", type=str, default="sim_transfer_cube_scripted")
    parser.add_argument("--train_seeds", type=str, default="0,1,2")
    parser.add_argument("--eval_seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--only_experiments", type=str, default="")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--train_aug_stage_epochs", type=str, default="")
    parser.add_argument("--eval_rollouts", type=int, default=50)
    parser.add_argument("--gate_baseline_success", type=float, default=0.98)
    parser.add_argument("--gate_vision_success", type=float, default=0.90)
    parser.add_argument("--python_bin", type=str, default=sys.executable)

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_seeds = parse_int_list(args.train_seeds)
    eval_seeds = parse_int_list(args.eval_seeds)

    if args.only_experiments:
        only_exp = set(parse_name_list(args.only_experiments))
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in only_exp}
        missing = sorted(only_exp - set(experiments.keys()))
        if missing:
            raise ValueError(f"unknown experiments: {missing}")
    else:
        experiments = EXPERIMENTS

    if args.skip_train and args.skip_eval:
        raise ValueError("skip_train and skip_eval cannot both be true")
    if args.eval_rollouts <= 0:
        raise ValueError("eval_rollouts must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    env = dict(**os.environ)

    records = []
    started_at = time.time()
    for exp_name, exp_cfg in experiments.items():
        for train_seed in train_seeds:
            ckpt_dir = output_root / exp_name / f"seed_{train_seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if not args.skip_train:
                trained_ckpt = ckpt_dir / "policy_best.ckpt"
                if trained_ckpt.exists() and not args.force_train:
                    print(f"[skip] existing checkpoint: {trained_ckpt}")
                else:
                    train_cmd = build_train_cmd(args.python_bin, args, ckpt_dir, train_seed, exp_cfg)
                    code, _ = run_cmd(train_cmd, repo_root, env, dry_run=args.dry_run)
                    if code != 0:
                        raise RuntimeError(f"training failed: exp={exp_name} seed={train_seed}")

            if args.skip_eval:
                continue
            if not (ckpt_dir / "policy_best.ckpt").exists() and not args.dry_run:
                raise FileNotFoundError(f"missing checkpoint for evaluation: {(ckpt_dir / 'policy_best.ckpt')}")

            for eval_seed in eval_seeds:
                for mode_name in MODE_SETTINGS.keys():
                    eval_cmd = build_eval_cmd(args.python_bin, args, ckpt_dir, eval_seed, mode_name)
                    code, raw = run_cmd(eval_cmd, repo_root, env, dry_run=args.dry_run)
                    if code != 0:
                        raise RuntimeError(
                            f"evaluation failed: exp={exp_name} train_seed={train_seed} "
                            f"eval_seed={eval_seed} mode={mode_name}"
                        )
                    if args.dry_run:
                        continue
                    success_rate, avg_return = parse_eval_metrics(raw)
                    records.append(
                        {
                            "experiment": exp_name,
                            "train_seed": int(train_seed),
                            "eval_seed": int(eval_seed),
                            "mode": mode_name,
                            "success_rate": float(success_rate),
                            "avg_return": float(avg_return),
                            "ckpt_dir": str(ckpt_dir),
                        }
                    )

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started_at, 3),
        "task_name": args.task_name,
        "output_root": str(output_root),
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "eval_rollouts": int(args.eval_rollouts),
        "gate": {
            "baseline_success_min": float(args.gate_baseline_success),
            "vision_success_min": float(args.gate_vision_success),
        },
        "experiments": experiments,
        "records": records,
    }

    if not args.dry_run:
        summary = aggregate_records(records, args.gate_baseline_success, args.gate_vision_success)
        result["summary"] = summary
        passing = [k for k, v in summary.items() if v["gate"]["pass_all"]]
        result["pass_all_experiments"] = passing
        best_exp = None
        best_value = -1.0
        for exp_name, exp_info in summary.items():
            val = exp_info["vision"]["success_rate"]["mean"]
            if val is None:
                continue
            if val > best_value:
                best_value = val
                best_exp = exp_name
        result["best_by_vision_success"] = best_exp

    json_path = output_root / "matrix_summary.json"
    json_path.write_text(json.dumps(result, indent=2))

    if not args.dry_run:
        csv_path = output_root / "matrix_records.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment",
                    "train_seed",
                    "eval_seed",
                    "mode",
                    "success_rate",
                    "avg_return",
                    "ckpt_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

    print(f"[done] wrote {json_path}")
    if not args.dry_run:
        print("[summary]")
        for exp_name in sorted(result["summary"].keys()):
            exp_info = result["summary"][exp_name]
            base = exp_info["baseline"]["success_rate"]["mean"]
            vis = exp_info["vision"]["success_rate"]["mean"]
            dyn = exp_info["dynamics"]["success_rate"]["mean"]
            gate_ok = exp_info["gate"]["pass_all"]
            print(
                f"  {exp_name}: baseline={base:.4f} vision={vis:.4f} dynamics={dyn:.4f} "
                f"pass_all={gate_ok}"
            )


if __name__ == "__main__":
    main()
