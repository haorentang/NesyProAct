"""
Main Executor Module - Coordinates planner and skill execution

Coordinates the execution flow:
1. Use Planner to decompose main goal into subtasks
2. For each subtask, execute using SkillExecutor
"""
import logging
from typing import Optional, Dict, Any, List
import dataclasses
import json
import os
from planner.planner import Planner, SubTask
from orchestrator.skill_executor import SkillExecutor
from browsergym.utils.obs import flatten_axtree_to_str

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SubgoalExecutionRecord:
    """Record of a subgoal execution for proactive grounding"""
    subgoal_id: int
    description: str
    axtree_after: str  # AXTree after subgoal execution (string format)
    grounded_elements: List[Dict[str, Any]]  # List of grounded elements with their predicates
    # Each element: {"element_id": str, "predicate": str, "action_type": str}
    yaml_action_sequence: Optional[List[Dict[str, Any]]] = None  # Full YAML action sequence if available


class MainExecutor:
    """
    Main executor that coordinates planner and skill execution
    """

    def __init__(
        self,
        planner: Planner,
        skill_executor: SkillExecutor
    ):
        """
        Initialize main executor

        Args:
            planner: Task planner for decomposition
            skill_executor: Skill executor for subtask execution
        """
        self.planner = planner
        self.skill_executor = skill_executor

        # Track subgoal execution history for proactive grounding
        self.subgoal_history: List[SubgoalExecutionRecord] = []
        self.task_goal_text: Optional[str] = None  # Store task goal text for skill reuse

    def execute_goal(
        self,
        goal_object: tuple[Dict[str, Any]],
        page_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a main goal by decomposing and executing subtasks

        Args:
            goal_object: Main goal in OpenAI message format
            page_state: Optional initial page state

        Returns:
            Execution result summary
        """
        logger.info("Starting goal execution...")

        # Get current page state (axtree and url) for planning
        current_axtree = None
        current_url = None


        obs = self.skill_executor._get_env_obs()

        axtree_object = obs.get("axtree_object", {})


        current_axtree = flatten_axtree_to_str(axtree_object)

        current_url = obs.get("url")

        # Extract task goal text for skill reuse
        task_goal_text = ""
        for msg in goal_object:
            if msg.get("type") == "text":
                task_goal_text += msg.get("text", "") + "\n"
        self.task_goal_text = task_goal_text.strip()

        # Step 1: Plan - decompose the goal into subgoals
        subtasks, obs = self.planner.plan(goal_object, axtree=current_axtree, url=current_url)
        if obs is not None:
            self.skill_executor.recent_obs = obs
        logger.info(f"Goal decomposed into {len(subtasks)} subtasks")

        results = {
            "total_subtasks": len(subtasks),
            "completed": 0,
            "failed": 0,
            "subtask_results": []
        }

        # Step 2: Execute each subtask
        while not self.planner.is_all_completed():
            current_subtask = self.planner.get_current_subtask()

            if current_subtask is None:
                logger.warning("No executable subtask found, but not all completed")
                break

            logger.info(f"Executing subtask {current_subtask.id}: {current_subtask.description}")

            # Verify precondition predicates before executing subtask
            precondition_met, precondition_message = self._verify_precondition_predicates(current_subtask)


            if not precondition_met:
                logger.warning(f"Precondition not met for subgoal {current_subtask.id} ({current_subtask.description}): {precondition_message}")
                logger.warning("Continuing execution; the step-wise verifiers will check the outcome.")
            else:
                logger.info(f"Precondition met for subtask {current_subtask.id} ({current_subtask.description}): {precondition_message}")

            # Execute subtask
            success = self.skill_executor.execute_subtask(
                current_subtask,
                page_state,
                task_goal=self.task_goal_text
            )

            # Update subtask status
            if success:
                self.planner.mark_subtask_completed(
                    current_subtask.id,
                    result="Execution successful"
                )
                results["completed"] += 1
            else:
                self.planner.mark_subtask_failed(
                    current_subtask.id,
                    error="Execution failed"
                )
                results["failed"] += 1

            results["subtask_results"].append({
                "id": current_subtask.id,
                "description": current_subtask.description,
                "success": success
            })

        # Task-level success is judged by an LLM evaluator over the final page, the action
        # history, and the agent's answer, following the ASI evaluation protocol
        # (autoeval/). Set NESY_EVAL=0 to skip it.
        results["evaluation"] = {}
        if os.environ.get("NESY_EVAL", "1") != "0":
            from autoeval.evaluator import (
                evaluate_task_execution,
                collect_action_history_from_memory,
                collect_response_from_memory
            )
            memory_entries = self.skill_executor.memory.get_all_entries()
            results["evaluation"] = evaluate_task_execution(
                intent=goal_object,
                action_history=collect_action_history_from_memory(memory_entries),
                response=collect_response_from_memory(memory_entries),
                final_obs=self.skill_executor._get_env_obs()
            )
            logger.info(f"Evaluation: {results['evaluation'].get('status', 'unknown')}")

        logger.info(f"Goal execution completed: {results['completed']}/{results['total_subtasks']} succeeded")

        return results



    def _verify_precondition_predicates(
        self,
        subtask: SubTask
    ) -> tuple[bool, str]:
        """
        Verify precondition predicates before executing a subgoal

        Checks three types of preconditions:
        1. previous_actions: Verify that certain action types exist in previous action sequences
        2. url: Verify that certain keywords exist in current URL
        3. axtree: Verify that certain keywords exist in current page axtree

        Args:
            subtask: The subtask to verify preconditions for

        Returns:
            Tuple of (all_met: bool, message: str)
        """
        try:
            # Check if precondition predicates exist
            if not subtask.precondition_predicates:
                return True, "No preconditions"

            precondition_predicates = subtask.precondition_predicates

            # 1. Verify previous_actions: Check if action types exist in previous action sequences
            previous_actions = precondition_predicates.get("previous_actions", [])
            if previous_actions:
                # Get all previous action sequences from memory entries
                all_previous_action_types = set()
                if self.skill_executor.memory and self.skill_executor.memory.entries:
                    for entry in self.skill_executor.memory.entries:
                        if entry.steps:
                            for step in entry.steps:
                                if step.action_type:
                                    all_previous_action_types.add(step.action_type)

                # Check if all required action types exist
                missing_actions = []
                for required_action in previous_actions:
                    if required_action not in all_previous_action_types:
                        missing_actions.append(required_action)

                if missing_actions:
                    error_msg = f"Previous actions not found: {missing_actions}. Found: {list(all_previous_action_types)}"
                    logger.warning(f"  ✗ {error_msg}")
                    return False, error_msg
                else:
                    logger.info(f"  ✓ Previous actions verified: {previous_actions}")

            # 2. Verify url: Check if keywords exist in current URL
            url_keywords = precondition_predicates.get("url", [])
            if url_keywords:
                # Get current URL from env observation
                current_url = None
                if self.skill_executor.env:
                    obs = self.skill_executor._get_env_obs()
                    if obs:
                        current_url = obs.get("url", "")

                if not current_url:
                    error_msg = "Cannot get current URL for verification"
                    logger.warning(f"  ✗ {error_msg}")
                    return False, error_msg

                # Check if all keywords exist in URL
                missing_keywords = []
                for keyword in url_keywords:
                    if keyword.lower() not in current_url.lower():
                        missing_keywords.append(keyword)

                if missing_keywords:
                    error_msg = f"URL keywords not found: {missing_keywords}. Current URL: {current_url}"
                    logger.warning(f"  ✗ {error_msg}")
                    return False, error_msg
                else:
                    logger.info(f"  ✓ URL keywords verified: {url_keywords}")

            # 3. Verify axtree: Check if keywords exist in current page axtree
            axtree_keywords = precondition_predicates.get("axtree", [])
            if axtree_keywords:
                # Get current axtree from env observation
                current_axtree_str = None
                if self.skill_executor.env:
                    obs = self.skill_executor._get_env_obs()
                    if obs:
                        axtree_object = obs.get("axtree_object", {})
                        if axtree_object:
                            current_axtree_str = flatten_axtree_to_str(axtree_object)

                if not current_axtree_str:
                    error_msg = "Cannot get current axtree for verification"
                    logger.warning(f"  ✗ {error_msg}")
                    return False, error_msg

                # Check if all keywords exist in axtree (case-insensitive)
                missing_keywords = []
                axtree_lower = current_axtree_str.lower()
                for keyword in axtree_keywords:
                    if keyword.lower() not in axtree_lower:
                        missing_keywords.append(keyword)

                if missing_keywords:
                    error_msg = f"Axtree keywords not found: {missing_keywords}"
                    logger.warning(f"  ✗ {error_msg}")
                    return False, error_msg
                else:
                    logger.info(f"  ✓ Axtree keywords verified: {axtree_keywords}")

            return True, "All preconditions met"

        except Exception as e:
            logger.warning(f"Failed to verify precondition predicates: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # On error, allow execution to proceed
            return True, f"Precondition verification error (allowing execution): {e}"
