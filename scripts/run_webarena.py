"""Run NesyProAct on WebArena tasks.

Each task is run in a fresh BrowserGym environment; induced skills persist across tasks in
skill_memory/<site>_<model>/ so later tasks can reuse them. Results go to
<output_dir>/<task_id>/{result.json, log.json}; result.json contains the LLM evaluation used to
compute the success rate (see scripts/compute_success_rate.py).

Usage (from the repository root, after generating configs with configs/generate_task_configs.py):
  python scripts/run_webarena.py --site shopping
  python scripts/run_webarena.py --site gitlab --task_ids 44 45 46
  python scripts/run_webarena.py --site reddit --start 0 --end 50
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import litellm  # noqa: E402
from main import create_executor, load_config_from_json  # noqa: E402

litellm.drop_params = True  # drop parameters a backbone does not support (e.g., temperature)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_configs(config_dir: Path, site: str, task_ids: Optional[List[int]] = None,
                 start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the task configs whose first site equals `site`, sorted by task_id."""
    configs = []
    for path in config_dir.glob("*.json"):
        config = load_config_from_json(str(path))
        sites = config.get("sites") or []
        if sites and sites[0] == site:
            configs.append(config)
    configs.sort(key=lambda c: int(c["task_id"]))
    if task_ids is not None:
        wanted = set(task_ids)
        configs = [c for c in configs if int(c["task_id"]) in wanted]
    return configs[start:end]


def run_task(config: Dict[str, Any], output_dir: Path, headless: bool) -> bool:
    """Run one task and save its result. Returns False if the run crashed."""
    task_id = config["task_id"]
    task_dir = output_dir / str(task_id)
    if (task_dir / "result.json").exists():
        logger.info(f"Task {task_id}: already done, skipping")
        return True
    task_dir.mkdir(parents=True, exist_ok=True)

    executor = None
    try:
        executor = create_executor(
            start_url=config["start_url"],
            require_login=config.get("require_login", False),
            site=config["sites"][0],
            headless=headless,
        )
        logger.info(f"Task {task_id}: {config['intent']}")
        start_time = time.time()
        results = executor.execute_goal(({"type": "text", "text": config["intent"]},))
        results["elapsed_s"] = round(time.time() - start_time, 1)

        with open(task_dir / "log.json", "w", encoding="utf-8") as f:
            json.dump(executor.skill_executor.memory.to_dict(), f, indent=2, ensure_ascii=False)
        with open(task_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Task {task_id}: {results['completed']}/{results['total_subtasks']} subtasks completed, "
                    f"evaluation={results.get('evaluation', {}).get('status')}")
        return True
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        with open(task_dir / "error.txt", "w", encoding="utf-8") as f:
            f.write(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        return False
    finally:
        if executor and executor.skill_executor.env:
            try:
                executor.skill_executor.env.close()
            except Exception as close_error:
                logger.warning(f"Task {task_id}: failed to close env: {close_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True,
                        help="WebArena site: shopping, shopping_admin, reddit, gitlab, map, wikipedia")
    parser.add_argument("--config_dir", default=str(REPO_ROOT / "configs" / "tasks"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--start", type=int, default=None, help="Start index into the site's sorted task list")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--model_name", default=None, help="Overrides $LLM_MODEL_NAME")
    parser.add_argument("--no_headless", action="store_true")
    args = parser.parse_args()

    if args.model_name:
        os.environ["LLM_MODEL_NAME"] = args.model_name
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    output_dir = Path(args.output_dir or REPO_ROOT / "results" / f"{args.site}_{model_name.replace('/', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = find_configs(Path(args.config_dir), args.site, args.task_ids, args.start, args.end)
    if not configs:
        logger.warning(f"No tasks found for site '{args.site}' in {args.config_dir}")
        return
    logger.info(f"Running {len(configs)} {args.site} tasks with {model_name} -> {output_dir}")

    crashed = sum(not run_task(c, output_dir, headless=not args.no_headless) for c in configs)
    logger.info(f"Finished {len(configs)} tasks ({crashed} crashed)")


if __name__ == "__main__":
    main()
