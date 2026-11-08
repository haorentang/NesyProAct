"""Run Online-Mind2Web tasks with NesyProAct on live websites.

Reuses create_executor()/execute_goal() and adds a step hook on the BrowserGym env to record
per-step screenshots and the action sequence required by the official WebJudge evaluator
(https://github.com/OSU-NLP-Group/Online-Mind2Web).

Usage (from the repository root):
  python scripts/run_online_mind2web.py --tasks Online_Mind2Web.json \
      [--output_dir results/om2w] [--max_steps 30] [--headless]

Task file format: JSON list of Online-Mind2Web records with at least
  task_id, website (start URL), confirmed_task (goal), level.
"""

import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_live_patch  # noqa: F401  (must precede any browsergym env creation)

from main import create_executor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StepBudgetExceeded(Exception):
    pass


def attach_recorder(env, shot_dir: Path, max_steps: int):
    """Wrap env.step to save screenshots + action log. Returns the log list."""
    shot_dir.mkdir(parents=True, exist_ok=True)
    action_log = []
    orig_step = env.step

    def recording_step(action):
        if len(action_log) >= max_steps:
            raise StepBudgetExceeded(f"exceeded {max_steps} env steps")
        out = orig_step(action)
        obs = out[0]
        idx = len(action_log)
        entry = {"step": idx, "action": str(action), "url": obs.get("url", "")}
        try:
            shot = obs.get("screenshot")
            if shot is not None:
                path = shot_dir / f"step_{idx:03d}.png"
                Image.fromarray(shot).save(path)
                entry["screenshot"] = path.name
        except Exception as e:
            entry["screenshot_error"] = str(e)
        action_log.append(entry)
        return out

    env.step = recording_step
    return action_log


def extract_final_response(action_log):
    """Last send_msg_to_user payload, if any (WebJudge uses it for info tasks)."""
    for entry in reversed(action_log):
        action = entry["action"]
        if "send_msg_to_user" in action:
            start = action.find("(")
            if start != -1:
                return action[start + 1:action.rfind(")")].strip("'\" ")
    return ""


def synthesize_final_response(goal, action_log, executor):
    """WebJudge expects a final answer for information-seeking tasks. When the agent sent no
    explicit answer, ask the same backbone to state the outcome from the trajectory. Such answers
    are marked as 'synthesized' in result.json."""
    import litellm
    trail = "\n".join(
        f"{e['step']}. {e['action'][:200]} [url: {e.get('url', '')[:120]}]"
        for e in action_log[-15:])
    try:
        last_obs = executor.skill_executor.recent_obs or {}
        from browsergym.utils.obs import flatten_axtree_to_str
        page = flatten_axtree_to_str(last_obs.get("axtree_object", {}))[:6000]
    except Exception:
        page = ""
    try:
        resp = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL_NAME", "gpt-4o"),
            temperature=0,
            messages=[{"role": "user", "content":
                f"A web agent worked on this task:\n{goal}\n\n"
                f"Its actions:\n{trail}\n\nFinal page (accessibility tree):\n{page}\n\n"
                "State the agent's final answer/result for the task in 1-3 sentences, "
                "based only on what the trajectory shows. If the task was not completed, "
                "say what state was reached."}])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"final response synthesis failed: {e}")
        return ""


def run_task(task, output_dir: Path, max_steps: int, headless: bool):
    task_id = str(task["task_id"])
    task_out = output_dir / task_id
    if (task_out / "result.json").exists():
        logger.info(f"Task {task_id}: done already, skip")
        return "skip"
    task_out.mkdir(parents=True, exist_ok=True)

    goal = task["confirmed_task"]
    start_url = task["website"]

    executor = None
    status = "error"
    result = None
    action_log = []
    start = time.time()
    try:
        executor = create_executor(
            start_url=start_url,
            use_browsergym=True,
            require_login=False,
            site="om2w",
            headless=headless,
        )
        env = executor.skill_executor.env
        if env is None:
            raise RuntimeError("browsergym env failed to initialize")
        action_log = attach_recorder(env, task_out / "screenshots", max_steps)

        result = executor.execute_goal(({"type": "text", "text": goal},))
        status = "completed"
    except StepBudgetExceeded:
        status = "truncated"
    except Exception as e:
        status = "error"
        (task_out / "error.txt").write_text(
            f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", encoding="utf-8")
        logger.error(f"Task {task_id}: {type(e).__name__}: {e}")
    finally:
        elapsed = time.time() - start
        if executor is not None:
            try:
                mem = executor.skill_executor.memory.to_dict()
                (task_out / "log.json").write_text(
                    json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            try:
                if executor.skill_executor.env is not None:
                    executor.skill_executor.env.close()
            except Exception:
                pass
        (task_out / "actions.json").write_text(
            json.dumps(action_log, indent=2, ensure_ascii=False), encoding="utf-8")
        final_response = extract_final_response(action_log)
        final_source = "agent"
        if not final_response and status == "completed" and executor is not None:
            final_response = synthesize_final_response(goal, action_log, executor)
            final_source = "synthesized"
        (task_out / "final_response.txt").write_text(final_response, encoding="utf-8")
        (task_out / "result.json").write_text(json.dumps({
            "task_id": task_id,
            "goal": goal,
            "start_url": start_url,
            "level": task.get("level"),
            "status": status,
            "final_response_source": final_source,
            "n_steps": len(action_log),
            "elapsed_s": round(elapsed, 1),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "executor_result": result,
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"Task {task_id}: {status} ({len(action_log)} steps, {elapsed:.0f}s)")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Overrides LLM_MODEL_NAME env var if given")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--task_timeout", type=int, default=900,
                        help="Hard wall-clock limit per task (heavy sites can hang env setup)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--single_index", type=int, default=None,
                        help="Internal: run exactly one task in-process (no timeout guard)")
    args = parser.parse_args()

    if args.model_name:
        os.environ["LLM_MODEL_NAME"] = args.model_name
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    output_dir = Path(args.output_dir or f"results/om2w_{model.replace('/', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[:args.limit]

    if args.single_index is not None:
        task = tasks[args.single_index]
        run_task(task, output_dir, args.max_steps, args.headless)
        return

    # parent mode: one subprocess per task, so a hung env setup (seen on
    # script-heavy sites) is bounded by task_timeout instead of stalling the batch
    import subprocess
    logger.info(f"Running {len(tasks)} tasks with model={model} -> {output_dir}")
    counts = {}
    for i, task in enumerate(tasks):
        tid = str(task["task_id"])
        if (output_dir / tid / "result.json").exists():
            logger.info(f"[{i + 1}/{len(tasks)}] task {tid}: done already, skip")
            counts["skip"] = counts.get("skip", 0) + 1
            continue
        logger.info(f"[{i + 1}/{len(tasks)}] task {tid}")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--tasks", args.tasks, "--output_dir", str(output_dir),
               "--max_steps", str(args.max_steps), "--single_index", str(i)]
        if args.headless:
            cmd.append("--headless")
        try:
            subprocess.run(cmd, cwd=os.getcwd(),
                           timeout=args.task_timeout,
                           # keep Windows from opening a console window per task
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            logger.error(f"Task {tid}: killed after {args.task_timeout}s wall-clock")
            task_out = output_dir / tid
            task_out.mkdir(parents=True, exist_ok=True)
            if not (task_out / "result.json").exists():
                (task_out / "result.json").write_text(json.dumps({
                    "task_id": tid, "goal": task["confirmed_task"],
                    "start_url": task["website"], "level": task.get("level"),
                    "status": "env_timeout", "n_steps": 0,
                    "elapsed_s": args.task_timeout,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "unknown"
        rj = output_dir / tid / "result.json"
        if rj.exists():
            try:
                status = json.loads(rj.read_text(encoding="utf-8")).get("status", "unknown")
            except Exception:
                pass
        counts[status] = counts.get(status, 0) + 1
        logger.info(f"Task {tid}: -> {status}")
    logger.info(f"Done: {counts}")


if __name__ == "__main__":
    main()
