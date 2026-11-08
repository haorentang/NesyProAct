"""
Skill executor: runs one subgoal.

1. Skill reuse: if a stored skill matches the subgoal (LLM-free URL match + reuse decision),
   replay it with every step re-grounded on the current page, then verify the outcome.
2. Otherwise (or if reuse fails), synthesize and execute a new action-query program with
   step-wise grounding and verification (SkillManager).
3. Successful new executions are distilled into reusable skills.
"""
import logging
import os
import re
from typing import Optional, Dict, Any, List

import litellm
from browsergym.utils.obs import flatten_axtree_to_str

from planner.planner import Planner, SubTask
from planner.verbs import extract_verbs_from_description, get_intents_and_roles_for_verbs
from memory.memory import Memory, MemoryEntry, ExecutionOutcome, ExecutionStep
from orchestrator.llm_judge import llm_judge_subtask
from orchestrator.skill_manager import SubtaskManager as SkillManager
from orchestrator.utils import (
    save_execution_experience, load_skill_from_mem, extract_action_from_response,
    find_element_bid_by_value, get_available_skills_with_url_match, decide_skill_reuse,
)
from orchestrator.prompts import (
    TEXT_GROUNDING_PROMPT, TEXT_GROUNDING_USER_PROMPT,
    SKILL_REPLAY_JUDGE_PROMPT, JUDGE_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    Executes a single subtask using the skill-based workflow
    """

    def __init__(
        self,
        memory: Memory,
        env: Optional[Any] = None,  # gymnasium.Env instance
        planner: Optional[Planner] = None,
        recent_obs: Optional[Dict[str, Any]] = None,
        site: str = "default",
        model_name: Optional[str] = None
    ):
        """
        Initialize skill executor

        Args:
            memory: Memory system for storing/retrieving experiences
            env: Optional browsergym environment (gymnasium.Env instance) for browser interaction
            planner: Optional task planner for task decomposition
            site: Site name used to name the skill-memory directory
            model_name: LLM model name (if None, reads from LLM_MODEL_NAME env var, defaults to "gpt-4o-mini")
        """
        # Get model_name from environment variable if not provided
        if model_name is None:
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.memory = memory
        self.env = env
        self.info_storage = []
        self.planner = planner
        self.model_name = model_name
        self.mem_dir = skill_memory_dir(site, model_name)
        self.recent_obs = recent_obs

    def execute_subtask(
        self,
        subtask: SubTask,
        page_state: Dict[str, Any],
        task_goal: Optional[str] = None
    ) -> bool:
        """
        Execute a subtask (skill reuse first, otherwise synthesize a new program)

        Args:
            subtask: The subtask to execute
            page_state: Current page state information
            task_goal: Optional overall task goal for skill reuse decision

        Returns:
            True if execution succeeded, False otherwise

        """
        # Extract verbs from description and fuse their roles and intents into the query
        description = subtask.description
        self._task_goal = task_goal
        verbs = extract_verbs_from_description(description)
        verb_info = get_intents_and_roles_for_verbs(verbs)

        # Build enhanced query with description, roles, and intents
        query_parts = [description]
        if verb_info["roles"]:
            query_parts.append(f"Verb roles: {', '.join(verb_info['roles'])}")
        if verb_info["allowed_intents"]:
            query_parts.append(f"Allowed intents: {', '.join(verb_info['allowed_intents'])}")
        if verb_info["forbidden_intents"]:
            query_parts.append(f"Forbidden intents: {', '.join(verb_info['forbidden_intents'])}")

        query = ". ".join(query_parts)

        logger.info(f"Executing subtask: {query}")

        # Initialize memory entry (will be filled based on execution path)
        memory_entry = MemoryEntry(
            query=description,
            steps=[],
            operator=None,
            outcome=ExecutionOutcome.FAILURE
        )


        # Initialize temporary storage for execution record
        self._last_execution_record = None

        # Get current URL before execution
        obs = self._get_env_obs()
        url_before = obs.get("url", "") if obs else ""

        # Set url_before in memory_entry
        memory_entry.url_before = url_before

        # Try skill reuse: get available skills (URL matched) and ask reuse agent
        skill_name_to_reuse = None
        skill_parameter_to_reuse = None
        if task_goal:
            available_skills = get_available_skills_with_url_match(self.mem_dir, url_before)
            if available_skills:
                reuse_result = decide_skill_reuse(
                    task_goal,
                    query,
                    available_skills,
                    model_name=self.model_name
                )
                if reuse_result:
                    skill_name_to_reuse, skill_parameter_to_reuse = reuse_result
        logger.debug(f"skill_name_to_reuse: {skill_name_to_reuse}, parameter: {skill_parameter_to_reuse}")
        # Execute based on reuse decision
        success = False
        steps = []

        if skill_name_to_reuse:
            logger.info(f"Reuse agent decided to use skill: {skill_name_to_reuse}, parameter: {skill_parameter_to_reuse}")
            memory_entry.operator = "skill_reuse"
            success, steps = self._execute_skill_with_steps(
                subtask,
                skill_name=skill_name_to_reuse,
                skill_parameter=skill_parameter_to_reuse
            )
            memory_entry.steps = steps

            if not success:
                logger.warning(f"Skill {skill_name_to_reuse} execution failed, retrying with planner plan")
                if url_before:
                    obs, _, _, _, _ = self._execute_action(f"goto('{url_before}')")
                skill_name_to_reuse = None

        if not skill_name_to_reuse or not success:
            # Execute using planner plan (generate new skill)
            memory_entry.operator = "planner_plan"
            success, steps = self._execute_with_planner_plan(subtask,query)
            memory_entry.steps = steps
        # Save YAML action sequence from execution record if available
        if self._last_execution_record:
            memory_entry.yaml_action_sequence = self._last_execution_record.yaml_action_sequence
            logger.info(f"Saved YAML action sequence with {len(memory_entry.yaml_action_sequence)} actions")

        if not success:
            logger.warning("Skill execution failed.")
            memory_entry.operator = "planner_plan_fallback"

        # Use LLM to judge if the subtask execution was successful
        # This should be called for all subtasks regardless of execution path
        # Get current observation from env if available
        obs_after = None
        if self.env:
            obs_after = self._get_env_obs()

        if not skill_name_to_reuse:
            skill_name, skill_description, skill_parameter = llm_judge_subtask(
                description, memory_entry.steps, page_state, obs_after, model_name=self.model_name
            )
        else:
            skill_name = skill_name_to_reuse
            skill_description = ""
            skill_parameter = []

        memory_entry.skill_name = skill_name
        memory_entry.skill_description = skill_description
        memory_entry.skill_parameter = skill_parameter
        if skill_name or skill_description:
            logger.info(f"Generated skill: name='{skill_name}', description='{skill_description}', parameters='{skill_parameter}'")

        # Store the execution in memory and persist newly induced skills
        if not skill_name_to_reuse and success and skill_name:
            save_execution_experience(skill_name, memory_entry, success, mem_dir=self.mem_dir)
        self.memory.add(memory_entry)
        return success

    def _execute_skill_with_steps(
        self,
        subtask: SubTask,
        skill_name: Optional[str] = None,
        skill_parameter: Optional[str] = None
    ) -> tuple[bool, List[ExecutionStep]]:
        """
        Execute skill steps by finding matching elements in current page axtree

        For each step:
        1. Extract action_type
        2. Find element in current page axtree with matching value
        3. Get the bid of that element
        4. Execute the action with that bid

        Args:
            subtask: The subtask to execute
            skill_name: Optional skill name to use (if not provided, will try to get from subtask if it has one)
            skill_parameter: Optional parameter value for fill actions (from reuse agent)

        Returns:
            tuple[bool, List[ExecutionStep]]: (success, steps) - True if all steps executed successfully, and list of execution steps
        """

        # Use provided skill_name or try to get from subtask (for backward compatibility)
        skill_name_to_use = skill_name

        # Load skill from memory
        skill_entry = load_skill_from_mem(skill_name_to_use, self.mem_dir)
        if not skill_entry:
            logger.warning(f"Skill {skill_name_to_use} not found in memory")
            return False, []
        steps = skill_entry["steps"]

        executed_steps = []

        # Process each step
        for step_idx, prev_step in enumerate(steps):
            action_type = prev_step.get("action_type")

            # Get current observation
            obs = self._get_env_obs()
            if obs is None:
                logger.warning("No env observation available")
                return False, executed_steps

            axtree_object = obs.get("axtree_object", {})
            axtree = flatten_axtree_to_str(axtree_object)

            # Get URL before action
            url_before = obs.get("url", "")

            # Get target value from previous step's target_features
            target_value = None
            target_features = prev_step.get("target_features")
            if target_features:
                target_value = target_features.get("value", "")

            # Find element in current page with matching value
            matched_bid = None
            if target_value and action_type in {'click', 'fill', 'hover', 'press',"select_option"}:
                matched_bid = find_element_bid_by_value(
                    axtree, target_value, prev_step, action_type
                )

                if not matched_bid:
                    logger.warning(f"Could not find element for {action_type} action, skill reuse failed")
                    return False, executed_steps

                if action_type == 'fill':
                    # For fill, prioritize skill_parameter from reuse agent, then extract from previous step
                    if not skill_parameter:
                        logger.warning(f"No parameter provided for fill action, skipping")
                        return False, executed_steps
                    text = skill_parameter
                    logger.info(f"Using parameter from reuse agent for fill action: {text}")

                    if matched_bid and text:
                        action_string = f"{action_type}('{matched_bid}', '{text}')"
                    else:
                        logger.warning(f"Could not find element or text for fill action, skipping")
                        return False, executed_steps
                elif action_type == 'press':
                    # Reuse the key recorded in the stored step (defaults to Enter)
                    key_match = re.search(r"press\(['\"][^'\"]+['\"],\s*['\"]([^'\"]+)['\"]\)", prev_step.get("action_str") or "")
                    key = key_match.group(1) if key_match else "Enter"
                    action_string = f"press('{matched_bid}', '{key}')"
                elif action_type == 'select_option':
                    previous_action_string = prev_step.get("action_str")
                    # Extract value from previous action string (e.g., select_option('12134', 'prize') -> 'prize')
                    if previous_action_string:
                        # Match select_option('bid', 'value') pattern
                        match = re.search(r"select_option\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", previous_action_string)
                        if match:
                            # Use the value from previous action string
                            previous_value = match.group(2)
                            action_string = f"select_option('{matched_bid}', '{previous_value}')"
                        else:
                            # Fallback to target_value if parsing fails
                            logger.warning(f"Failed to extract value from previous action string, returning")
                            return False, executed_steps
                else:
                    action_string = f"{action_type}('{matched_bid}')"
            elif action_type in {'go_back', 'go_forward', 'noop'}:
                action_string = f"{action_type}()"
            elif action_type == 'store' or action_type == 'send_msg_to_user':
                if self.info_storage:
                    formatted_info = "\n".join([f"- {info}" for info in self.info_storage])
                else:
                    formatted_info = "No extra information available."
                logger.info("Using TEXT_GROUNDING for send_msg_to_user or store action")

                # Get task goal from planner if available, otherwise use subtask description
                task_goal = subtask.description
                if self.planner and hasattr(self.planner, 'current_task'):
                    task_goal = self.planner.current_task

                user_prompt = TEXT_GROUNDING_USER_PROMPT.format(
                    task_goal=task_goal,
                    subtask=subtask.description,
                    extra_info=formatted_info,
                    ax_tree=axtree
                )
                messages = [
                    {"role": "system", "content": TEXT_GROUNDING_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

                response = litellm.completion(
                    api_key=os.environ.get("LITELLM_API_KEY"),
                    base_url=os.environ.get("LITELLM_BASE_URL"),
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )
                response_text = response.choices[0].message.content
                message = extract_action_from_response(response_text)

                if not message:
                    logger.error(f"Failed to extract message from text grounding response: {response_text}")
                    break

                # Remove quotes if present
                message = message.strip("'\"")
                if action_type == 'send_msg_to_user':
                    action_string = f"send_msg_to_user('{message}')"
                else: # Store the extracted information
                    action_string = f"store('{message}')"
                    self.info_storage.append(message)
                    continue

            else:
                logger.warning(f"Unknown action type: {action_type}")
                action_string = f"{action_type}()"

            # Execute action
            logger.info(f"Executing action [{step_idx}]: {action_string}")
            obs_after, reward, terminated, truncated, info = self._execute_action(action_string)

            # Get URL after action
            url_after = obs_after.get("url", "") if obs_after else url_before
            url = url_after if url_after else url_before

            # Get axtree after action execution

            axtree_after_object = obs_after.get("axtree_object", {}) if obs_after else {}
            if not axtree_after_object:
                logger.warning("No accessibility tree after action execution; skill reuse failed")
                return False, executed_steps
            axtree_after_str = flatten_axtree_to_str(axtree_after_object)

            # Check for errors in observation
            success = True
            error_msg = None
            if obs_after and "last_action_error" in obs_after and obs_after["last_action_error"]:
                success = False
                error_msg = obs_after["last_action_error"]
                logger.warning(f"Action execution error: {error_msg}")
                return False, executed_steps

            # Extract information from prev_step for ExecutionStep
            semantic_target = prev_step.get("semantic_target")
            target_features = prev_step.get("target_features")
            action_context = prev_step.get("action_context")
            action_reason = prev_step.get("action_reason", f"Skill reuse: {skill_name}")

            # Use matched_bid as original_bid (or fallback to prev_step's original_bid)
            original_bid = str(matched_bid) if matched_bid else prev_step.get("original_bid")

            # Create ExecutionStep
            step = ExecutionStep(
                step_index=step_idx,
                action_type=action_type,
                action_str=action_string,
                semantic_target=semantic_target,
                target_features=target_features,
                original_bid=original_bid,
                action_reason=action_reason,
                action_context=action_context,
                error_msg=error_msg,
                url=url,
                axtree=axtree_after_str,
            )
            executed_steps.append(step)

            if not success:
                logger.warning(f"Action execution failed at step {step_idx}")
                return False, executed_steps

            # Check if episode is done
            if terminated or truncated:
                logger.info(f"Episode done (terminated: {terminated}, truncated: {truncated}) at step {step_idx + 1}")
                return terminated, executed_steps

        if not executed_steps:
            logger.warning("Skill replay executed no steps")
            return False, executed_steps

        logger.info("Judging if subgoal has been reached")
        user_prompt = JUDGE_USER_PROMPT.format(
            executed_subgoal=subtask.description,
            ax_tree=axtree_after_str,
            action_history="\n".join([step.action_str for step in executed_steps])
        )
        messages = [
            {"role": "system", "content": SKILL_REPLAY_JUDGE_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL"),
            model=self.model_name,
            messages=messages,
            temperature=0.3,
        )
        response_text = response.choices[0].message.content

        # Extract success or failure from the response
        judge_result = extract_action_from_response(response_text) or ""
        is_success = judge_result.strip().lower() == "success"

        return is_success, executed_steps

    def _execute_with_planner_plan(
        self,
        subtask: SubTask,
        query: str
    ) -> tuple[bool, List[ExecutionStep]]:
        """
        Execute subtask using SkillManager (orchestrator) execution

        Args:
            subtask: The subtask to execute
            query: Subtask description fused with verb roles and intents

        Returns:
            Tuple of (success: bool, steps: List[ExecutionStep])
        """
        # Check if env is available
        if self.env is None:
            logger.warning("No env available for planner plan execution")
            return False, []

        # Get current observation from env and extract axtree_object
        obs = self._get_env_obs()
        if obs is None:
            logger.warning("Cannot get observation from env, cannot execute planner plan")
            return False, []

        axtree_object = obs.get("axtree_object", {})

        # Convert axtree_object to string
        try:
            ax_tree_str = flatten_axtree_to_str(axtree_object)
        except Exception as e:
            logger.error(f"Failed to flatten axtree: {e}")
            return False, []

        # The overall task goal fills the prompts' "Overall goal" slot (falls back to the subtask description)
        overall_goal = getattr(self, "_task_goal", None) or subtask.description

        # Create SkillManager instance
        skill_manager = SkillManager(
            env=self.env,
            task=overall_goal,
            subtask=query,
            info_storage=self.info_storage,
            ax_tree=ax_tree_str,
            model_name=self.model_name
        )


        # Execute using SkillManager
        logger.info(f"Executing subtask using SkillManager: {subtask.description}")
        try:
            success, env_after, executed_actions, targeted_element_history, executed_action_strings,ax_tree_history, url_history, info_storage, terminated = skill_manager.execute()
            self.recent_obs = skill_manager.recent_obs
        except Exception as e:
            logger.error(f"SkillManager execution failed: {e}")
            return False, []

        self.info_storage = info_storage
        # Convert executed_actions, targeted_element_history, ax_tree_history, url_history to ExecutionStep list
        executed_steps = []

        # Process each executed action
        # Note: ax_tree_history includes initial axtree at index 0, so step_idx corresponds to:
        # - axtree_before: ax_tree_history[step_idx] (initial axtree for step 0)
        # - axtree_after: ax_tree_history[step_idx + 1] (axtree after executing step step_idx)
        for step_idx, action_obj in enumerate(executed_actions):
            action_type = action_obj.get("action", "unknown")
            intent_label = action_obj.get("intent", "")
            hint = action_obj.get("hint", "")

            # Get targeted element for this step
            targeted_element = targeted_element_history[step_idx] if step_idx < len(targeted_element_history) else {}
            element_id = targeted_element.get("id", "")

            action_string = executed_action_strings[step_idx]

            # Get axtree after this step execution
            # ax_tree_history[0] is initial axtree, ax_tree_history[step_idx+1] is axtree after step step_idx
            axtree_after_str = ax_tree_history[step_idx] if step_idx < len(ax_tree_history) else None

            # Get URL for this step
            url = url_history[step_idx] if step_idx < len(url_history) else None

            if not axtree_after_str:
                logger.warning("Missing accessibility tree for an executed step")
                return False, executed_steps
            # Extract bid from element_id
            original_bid = str(element_id) if element_id else None

            # Extract semantic information from targeted_element
            semantic_target = None
            target_features = None
            action_context = None

            if targeted_element:
                # Build target_features from targeted_element
                target_features = {
                    "role": targeted_element.get("role", ""),
                    "value": targeted_element.get("value", ""),
                    "attributes": targeted_element.get("attributes", []),
                }
                # Use hint as semantic_target if available
                semantic_target = hint if hint else None
                # Use context from targeted_element
                action_context = targeted_element.get("context", {})

            step = ExecutionStep(
                step_index=step_idx,
                action_type=action_type,
                action_str=action_string,
                semantic_target=semantic_target,
                target_features=target_features,
                original_bid=original_bid,
                action_reason=f"{intent_label if intent_label else 'Action executed'}",
                action_context=action_context,
                error_msg=None,
                url=url,
                axtree=axtree_after_str,
            )
            executed_steps.append(step)


        logger.info(f"SkillManager execution completed: success={success}, actions={len(executed_actions)}, terminated={terminated}")
        logger.info(f"Executed {len(executed_steps)} steps")
        if not success:
            logger.warning(f"SkillManager execution failed")

        # Store action sequence for later retrieval (if available)
        # Note: New version doesn't return execution_record, so we store action sequence from skill_manager
        if hasattr(skill_manager, 'action_sequence') and skill_manager.action_sequence:
            # Create a simple execution record structure for compatibility
            # This is used in execute_subtask to save yaml_action_sequence
            # New skill_manager doesn't have SkillManagerExecutionRecord, so create a simple object
            class SimpleExecutionRecord:
                def __init__(self, yaml_action_sequence, execution_steps, step_wise_url):
                    self.yaml_action_sequence = yaml_action_sequence
                    self.execution_steps = execution_steps
                    self.step_wise_url = step_wise_url

            self._last_execution_record = SimpleExecutionRecord(
                yaml_action_sequence=skill_manager.action_sequence,
                execution_steps=[],  # Empty since we don't have detailed step records
                step_wise_url=url_history
            )
        else:
            self._last_execution_record = None


        return success, executed_steps

    def _execute_action(self, action_string: str) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs_after, reward, terminated, truncated, info = self.env.step(action_string)
        self.recent_obs = obs_after
        return obs_after, reward, terminated, truncated, info

    def _get_env_obs(self) -> Optional[Dict[str, Any]]:
        """
        Get current observation from env using noop action

        Returns:
            Observation dictionary or None if env is not available
        """
        if self.env is None:
            return None
        if self.recent_obs is not None:
            return self.recent_obs
        else:
            try:
                # Use noop action to get current observation
                obs, _, _, _, _ = self.env.step("noop")
                return obs
            except Exception as e:
                logger.error(f"Failed to get observation from env: {e}")
                return None


def skill_memory_dir(site: str, model_name: str) -> str:
    """Directory holding induced skills for a site/backbone pair (overridable with NESY_SKILL_DIR)."""
    root = os.environ.get("NESY_SKILL_DIR", "skill_memory")
    return os.path.join(root, f"{site}_{model_name.replace('/', '_')}")
