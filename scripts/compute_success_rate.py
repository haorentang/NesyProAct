"""Compute the task success rate from run_webarena.py outputs.

A task counts as successful when the LLM evaluator (autoeval/, ASI protocol) returns status
"success" in result.json. Tasks that crashed (error.txt, no result.json) count as failures.

Usage:
  python scripts/compute_success_rate.py results/shopping_gpt-4o
  python scripts/compute_success_rate.py results/*_gpt-4o
"""
import json
import sys
from pathlib import Path


def is_success(result: dict) -> bool:
    evaluation = result.get("evaluation") or {}
    if "success" in evaluation:
        return evaluation["success"] is True
    return str(evaluation.get("status", "")).lower() == "success"


def summarize(run_dir: Path):
    task_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    successes = 0
    for d in task_dirs:
        result_file = d / "result.json"
        if result_file.exists():
            successes += is_success(json.loads(result_file.read_text(encoding="utf-8")))
    return successes, len(task_dirs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    total_s = total_n = 0
    for arg in sys.argv[1:]:
        s, n = summarize(Path(arg))
        total_s, total_n = total_s + s, total_n + n
        print(f"{arg}: {s}/{n} = {100 * s / max(n, 1):.1f}%")
    if len(sys.argv) > 2:
        print(f"Overall: {total_s}/{total_n} = {100 * total_s / max(total_n, 1):.1f}%")


if __name__ == "__main__":
    main()
