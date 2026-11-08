"""
NesyProAct entry point.

Builds the agent (planner + skill executor) on top of a BrowserGym environment and runs one task:
1. The planner decomposes the task into subgoals.
2. Each subgoal is executed by reusing a verified skill or by synthesizing a new
   action-query program with step-wise grounding and verification.

Usage:
    python main.py path/to/task_config.json
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import gymnasium as gym
import browsergym.core  # noqa: F401  (registers the browsergym/openended environment)

from planner.planner import Planner
from memory.memory import Memory
from orchestrator.skill_executor import SkillExecutor
from orchestrator.main_executor import MainExecutor
from auto_login import login_site_with_env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """Load a task configuration (WebArena format) from a JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_executor(
    start_url: Optional[str] = None,
    use_browsergym: bool = True,
    require_login: bool = False,
    site: str = "default",
    model_name: Optional[str] = None,
    headless: bool = True,
    page_timeout_ms: int = 60000,
) -> MainExecutor:
    """
    Create the NesyProAct executor.

    Args:
        start_url: Starting URL of the task
        use_browsergym: Whether to create a BrowserGym environment
        require_login: Whether to log in before the task starts (WebArena sites)
        site: Site name; used for login and to name the skill-memory directory
        model_name: LLM backbone (LiteLLM model name); defaults to $LLM_MODEL_NAME or gpt-4o-mini
        headless: Run the browser headless
        page_timeout_ms: Timeout for loading the start page
    """
    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    planner = Planner(model_name=model_name, max_subtasks=10, site=site)
    memory = Memory()

    env = None
    obs = None
    if use_browsergym and start_url:
        # Use a longer timeout than BrowserGym's default when loading the start page
        from browsergym.core.task import OpenEndedTask

        def _setup_with_timeout(self, page):
            page.goto(self.start_url, timeout=page_timeout_ms)
            return self.goal, {}
        OpenEndedTask.setup = _setup_with_timeout

        env = gym.make(
            "browsergym/openended",
            disable_env_checker=True,
            headless=headless,
            wait_for_user_message=False,
            timeout=page_timeout_ms,
            task_kwargs={"start_url": start_url, "goal": None},
        )
        obs, _ = env.reset()
        if require_login:
            login_site_with_env(env, site)
        obs, _, _, _, _ = env.step(f"goto('{start_url}')")
        logger.info(f"BrowserGym environment initialized with URL: {start_url}")

    skill_executor = SkillExecutor(
        memory=memory,
        env=env,
        planner=planner,
        recent_obs=obs,
        site=site,
        model_name=model_name,
    )
    return MainExecutor(planner=planner, skill_executor=skill_executor)


def run_from_config(config_path: str) -> Dict[str, Any]:
    """Run a single task from a WebArena-style JSON config."""
    config = load_config_from_json(config_path)
    sites = config.get("sites") or ["default"]
    executor = create_executor(
        start_url=config.get("start_url"),
        require_login=config.get("require_login", False),
        site=sites[0],
        headless=os.getenv("HEADLESS", "1") != "0",
    )
    goal_object = ({"type": "text", "text": config.get("intent")},)
    logger.info(f"Task {config.get('task_id', 'unknown')}: {config.get('intent')}")
    try:
        results = executor.execute_goal(goal_object)
    finally:
        if executor.skill_executor.env:
            executor.skill_executor.env.close()

    logger.info(f"Subtasks: {results['completed']}/{results['total_subtasks']} completed")
    for result in results['subtask_results']:
        logger.info(f"[{'OK' if result['success'] else 'FAIL'}] {result['id']}: {result['description']}")
    logger.info(f"Evaluation: {results.get('evaluation')}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")):
        logger.warning("Neither OPENAI_API_KEY nor LITELLM_API_KEY is set.")
    run_from_config(sys.argv[1])
