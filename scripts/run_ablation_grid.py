from __future__ import annotations
import argparse
import csv
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

def run_cmd(cmd):
    print(">>>", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    print(out)
    return out

def parse_json_from_stdout(stdout: str):
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("Cannot find JSON in stdout")
    return json.loads(stdout[start:end+1])

def mode_robustness(args):
    results = []
    for deg in ["gaussian_noise", "motion_blur", "jpeg"]:
        for sev in [1,2,3,4,5]:
            cmd = [
                "python", "-m", "src.evaluate",
                "--config", args.config,
                "--ckpt", args.ckpt,
                "--eval_degradation", deg,
                "--eval_severity", str(sev),
            ]
            stdout = run_cmd(cmd)
            js = parse_json_from_stdout(stdout)
            results.append({
                "degradation": deg,
                "severity": sev,
                "miou": js["miou"],
                "pixel_acc": js["pixel_acc"],
                "loss": js["loss"],
            })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robustness_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["degradation","severity","miou","pixel_acc","loss"])
        writer.writeheader()
        writer.writerows(results)
    # 画图
    plt.figure(figsize=(7,5))
    for deg in ["gaussian_noise", "motion_blur", "jpeg"]:
        xs = [r["severity"] for r in results if r["degradation"] == deg]
        ys = [r["miou"] for r in results if r["degradation"] == deg]
        plt.plot(xs, ys, marker="o", label=deg)
    plt.xlabel("Severity")
    plt.ylabel("mIoU")
    plt.title("Robustness Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_curve.png", dpi=180)
    print("Saved:", csv_path, out_dir / "robustness_curve.png")

def mode_compare_runs(args):
    """
    汇总多个 run 下的 clean + 固定退化条件评估结果，用于比较 baseline vs deg-aware / 不同 p。
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.runs:
        # 格式：name=ckpt_path
        if "=" not in item:
            raise ValueError(f"Invalid --runs item: {item}, expected name=ckpt_path")
        name, ckpt = item.split("=", 1)
        for deg, sev in [(None, None), ("gaussian_noise", 3), ("motion_blur", 3), ("jpeg", 3)]:
            cmd = ["python", "-m", "src.evaluate", "--config", args.config, "--ckpt", ckpt]
            if deg is not None:
                cmd += ["--eval_degradation", deg, "--eval_severity", str(sev)]
            stdout = run_cmd(cmd)
            js = parse_json_from_stdout(stdout)
            rows.append({
                "run": name,
                "degradation": deg or "clean",
                "severity": 0 if sev is None else sev,
                "miou": js["miou"],
                "pixel_acc": js["pixel_acc"],
                "loss": js["loss"],
            })
    with open(out_dir / "compare_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run","degradation","severity","miou","pixel_acc","loss"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_dir / "compare_runs.csv")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["robustness", "compare_runs"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None, help="用于 robustness 模式")
    p.add_argument("--runs", nargs="*", default=[], help="用于 compare_runs 模式，格式 name=ckpt_path")
    p.add_argument("--out_dir", default="./analysis_outputs")
    args = p.parse_args()

    if args.mode == "robustness":
        if not args.ckpt:
            raise ValueError("--ckpt is required for robustness mode")
        mode_robustness(args)
    else:
        if not args.runs:
            raise ValueError("--runs is required for compare_runs mode")
        mode_compare_runs(args)

if __name__ == "__main__":
    main()
