"""Convert run_online_mind2web.py outputs into the WebJudge input layout.

Produces <out_dir>/<task_id>/{result.json, trajectory/*.png} with
result.json = {task_id, task, final_result_response, action_history}, which can be scored with the
official WebJudge evaluator (https://github.com/OSU-NLP-Group/Online-Mind2Web).

Usage:
  python scripts/convert_for_webjudge.py --tasks Online_Mind2Web.json \
      --run_dir results/om2w_gpt-4o --out_dir results/om2w_gpt-4o_webjudge
"""
import argparse
import json
import re
import shutil
from pathlib import Path


def extract_final_from_actions(actions):
    for a in reversed(actions):
        m = re.search(r"send_msg_to_user\(\s*(['\"])(.*)\1\s*\)", a, re.DOTALL)
        if m:
            return m.group(2)
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, help="Online-Mind2Web task file (JSON list)")
    parser.add_argument("--run_dir", required=True, help="Output directory of run_online_mind2web.py")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    tasks = {t["task_id"]: t for t in json.loads(Path(args.tasks).read_text(encoding="utf-8"))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tdir in sorted(p for p in Path(args.run_dir).iterdir() if p.is_dir()):
        task_id = tdir.name
        actions_file = tdir / "actions.json"
        if not actions_file.exists() or task_id not in tasks:
            continue
        log = json.loads(actions_file.read_text(encoding="utf-8"))
        actions = [e["action"] for e in log]
        shots = [tdir / "screenshots" / e["screenshot"] for e in log if "screenshot" in e]
        shots = [s for s in shots if s.exists()]
        final = ""
        fr = tdir / "final_response.txt"
        if fr.exists():
            final = fr.read_text(encoding="utf-8").strip()
        final = final or extract_final_from_actions(actions)

        traj = out_dir / task_id / "trajectory"
        traj.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(shots):
            shutil.copyfile(src, traj / f"{i}_full_screenshot.png")
        (out_dir / task_id / "result.json").write_text(json.dumps({
            "task_id": task_id,
            "task": tasks[task_id]["confirmed_task"],
            "final_result_response": final,
            "action_history": actions,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  {task_id}: {len(actions)} actions, {len(shots)} screenshots")


if __name__ == "__main__":
    main()
