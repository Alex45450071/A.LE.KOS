import json
import os
import re
import subprocess
import sys
from datetime import datetime


ROOT = r"c:\Users\alex_\Desktop\ALEKOS"
TASK3 = os.path.join(ROOT, "task3.py")
EVAL = os.path.join(ROOT, "eval_v3.py")
OUT = os.path.join(ROOT, "runs", "task3_model_sweep_results.json")
MODELS = ["cv", "ca", "kf", "ctrv", "hybrid"]


def run_cmd(command, env=None):
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_eval(stdout):
    ade_match = re.search(r"Mean ADE:\s*([0-9]*\.?[0-9]+)", stdout)
    fde_match = re.search(r"Mean FDE:\s*([0-9]*\.?[0-9]+)", stdout)
    med_match = re.search(r"Median ADE:\s*([0-9]*\.?[0-9]+)", stdout)
    if not ade_match:
        raise RuntimeError("Could not parse Mean ADE from eval output.")
    return {
        "mean_ade": float(ade_match.group(1)),
        "mean_fde": float(fde_match.group(1)) if fde_match else None,
        "median_ade": float(med_match.group(1)) if med_match else None,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    all_results = []

    for model in MODELS:
        print(f"\n=== Running model: {model} ===")
        env = os.environ.copy()
        env["TASK3_MODEL"] = model

        code, out, err = run_cmd([sys.executable, TASK3], env=env)
        if code != 0:
            raise RuntimeError(f"task3.py failed for model={model}\nSTDERR:\n{err}\nSTDOUT:\n{out}")

        code, eval_out, eval_err = run_cmd([sys.executable, EVAL], env=env)
        if code != 0:
            raise RuntimeError(
                f"eval_v3.py failed for model={model}\nSTDERR:\n{eval_err}\nSTDOUT:\n{eval_out}"
            )

        metrics = parse_eval(eval_out)
        result = {
            "model": model,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        all_results.append(result)
        fde_str = f"{metrics['mean_fde']:.3f}" if metrics["mean_fde"] is not None else "n/a"
        print(f"model={model} | ADE={metrics['mean_ade']:.3f} | FDE={fde_str}")

    winner = min(all_results, key=lambda x: x["metrics"]["mean_ade"])
    payload = {"results": all_results, "winner": winner}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== WINNER ===")
    winner_fde = (
        f"{winner['metrics']['mean_fde']:.3f}"
        if winner["metrics"]["mean_fde"] is not None
        else "n/a"
    )
    print(
        f"model={winner['model']} | ADE={winner['metrics']['mean_ade']:.3f} | "
        f"FDE={winner_fde}"
    )
    print(f"Saved sweep results to: {OUT}")


if __name__ == "__main__":
    main()
