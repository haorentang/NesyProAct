from gymnasium import Env
import litellm
import json
import os
import logging

from interface.intents import INTENT_LABEL
from interface.state_registry import retrieve_click, retrieve_fill, retrieve_hover, retrieve_press, retrieve_select_option, format_candidates_for_llm
from interface.utils import extract_all_elements_with_context
from browsergym.utils.obs import flatten_axtree_to_str
from orchestrator.utils import extract_and_parse_yaml_action_sequence, validate_yaml_action_sequence, extract_action_from_response, extract_targeted_element_id_from_action_string
from orchestrator.prompts import (
    ACTION_PLAN_PROMPT, ACTION_PLAN_USER_PROMPT,
    GROUNDING_PROMPT, GROUNDING_USER_PROMPT,
    RAW_GROUNDING_PROMPT, RAW_GROUNDING_USER_PROMPT,
    TEXT_GROUNDING_PROMPT, TEXT_GROUNDING_USER_PROMPT,
    SUBGOAL_JUDGE_PROMPT, JUDGE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

class SubtaskManager:
    def __init__(self, env: Env, task: str, subtask: str, ax_tree: str, info_storage: list[str] = None, max_actions: int = int(os.environ.get("NESY_MAX_ACTIONS", 3)), model_name: str = "gpt-4o-mini"):
        self.env = env
        self.recent_obs = None
        self.task = task
        self.subtask = subtask
        self.init_ax_tree = ax_tree
        self.current_ax_tree = ax_tree
        self.model_name = model_name
        self.info_storage = info_storage if info_storage is not None else []
        self.max_actions = max_actions
        self.litellm_api_key = os.environ.get("LITELLM_API_KEY")
        self.litellm_base_url = os.environ.get("LITELLM_BASE_URL")

    def _get_action_sequence_for_current_page(self, executed_action_strings: list[str], max_attempts: int = 2) -> tuple[bool, str, list[dict]]:
        """
        Get the action sequence for the current page from the model.
        Args:
            max_attempts: The maximum number of attempts to get the action sequence.
        Returns:
        A tuple containing a boolean indicating whether the action sequence is valid, the workflow YAML string, and a list of action objects in the format:
        {
            "action": <action_type>,
            "intent": <intent_label>,
            "hint": <action_hint>
        }
        """
        # Format intent labels for the system prompt
        intent_labels_str = "\n".join([
            f"- {label}: {info['description']} (Allowed actions: {', '.join(info['allowed_actions'])})"
            for label, info in INTENT_LABEL.items()
        ])
        system_prompt = ACTION_PLAN_PROMPT.format(overall_goal=self.task, subtask=self.subtask, INTENT_LABELS=intent_labels_str, action_history=executed_action_strings)
        
        # Initialize message history with system prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Initial user prompt
        initial_user_prompt = ACTION_PLAN_USER_PROMPT.format(
            overall_goal=self.task,
            subtask=self.subtask,
            action_history="None",
            current_ax_tree=self.current_ax_tree
        )
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Attempting to get action sequence for current page (attempt {attempt}/{max_attempts})")
                
                # Add user prompt to messages (only on first attempt)
                # On subsequent attempts, the conversation already contains error feedback from previous attempts
                if attempt == 1:
                    messages.append({"role": "user", "content": initial_user_prompt})
                
                # 1. get the response from the model
                response = litellm.completion(
                    api_key=self.litellm_api_key,
                    base_url=self.litellm_base_url,
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )

                response_text = response.choices[0].message.content
                
                # Add assistant response to message history
                messages.append({"role": "assistant", "content": response_text})
                
                # 2. extract and parse the YAML action sequence from the response
                action_sequence = extract_and_parse_yaml_action_sequence(response_text)
                
                # 3. validate the YAML action sequence
                is_valid, error_message = validate_yaml_action_sequence(action_sequence)
                
                if is_valid:
                    logger.info(f"Successfully generated valid action sequence with {len(action_sequence)} actions")
                    return True, response_text, action_sequence
                else:
                    logger.warning(f"Invalid action sequence (attempt {attempt}/{max_attempts}): {error_message}")
                    
                    if attempt < max_attempts:
                        # Add error message to the conversation for next attempt
                        error_feedback = f"The generated action sequence is invalid. Error: {error_message}\n\nPlease generate a corrected action sequence that addresses this error."
                        messages.append({"role": "user", "content": error_feedback})
                        logger.info("Added error feedback to conversation, retrying...")
                    else:
                        # Return False with the last response even if invalid
                        return False, response_text, action_sequence
                
            except Exception as e:
                logger.error(f"Failed to get action sequence (attempt {attempt}/{max_attempts}): {e}")
                
                if attempt < max_attempts:
                    # Add error feedback about the exception
                    error_feedback = f"An error occurred while processing the action sequence: {str(e)}\n\nPlease generate a corrected action sequence."
                    messages.append({"role": "user", "content": error_feedback})
                    logger.info("Added exception feedback to conversation, retrying...")
                    continue
                else:
                    # Return False with empty values on final failure
                    return False, "", []
        
        # If we get here, all attempts failed
        return False, "", []

    def execute(self) -> tuple[bool, Env, list[dict], list[dict], list[str], list[str], list[str], list[str], bool]:
        """
        Execute the subtask.
        Returns:
        A tuple containing:
        - bool: True if the subtask was executed successfully, False otherwise
        - Env: The environment after executing the subtask
        - list[dict]: The executed actions in the form:
        {
            "action": <action_type>,
            "intent": <intent_label>,
            "hint": <action_hint>
        }
        - list[dict]: The targeted element history in the form:
        {
            "id": <element_id>,
            "role": <element_role>,
            "value": <element_value>,
            "attributes": <element_attributes>,
            "context": <element_context>
        }
        - list[str]: The action string history during execution
        - list[str]: The ax tree history during execution
        - list[str]: The URL history after executing
        - list[str]: The updated information storage list with newly extracted info
        - bool: Only set to True if the environment was terminated
        """
        executed_actions = []  # list[dict] - action objects
        executed_action_strings = []
        targeted_element_history = []
        ax_tree_history = []
        url_history = []

        completed = False
        current_action_length = 0
        terminated = False
        truncated = False

        while not completed and current_action_length < self.max_actions:
            # get action sequence from current ax tree
            is_valid, action_sequence_str, action_sequence = self._get_action_sequence_for_current_page(executed_action_strings)
            if not is_valid or not action_sequence:
                logger.error("Failed to get valid action sequence for current page")
                return False, self.env, executed_actions, targeted_element_history, executed_action_strings, ax_tree_history, url_history, self.info_storage, terminated
            
            # for each action in the action sequence:
            for action in action_sequence:
                if current_action_length >= self.max_actions:
                    break
                    
                action_type = action.get('action')
                intent_label = action.get('intent')
                hint = action.get('hint')
                action_string = ""
                elements = None
                candidates = None

                # determine if the action needs grounding
                # if need grounding, retrieve elements using state_registry.retrieve_..., use hint as query
                if action_type in {'click', 'fill', 'hover', 'press', 'select_option'}:
                    # Extract elements from current ax_tree
                    elements = extract_all_elements_with_context(self.current_ax_tree)
                    
                    # Retrieve candidates based on action type
                    if action_type == 'click':
                        candidates = retrieve_click(elements, intent_label, query=hint, topk=10)
                    elif action_type == 'fill':
                        candidates = retrieve_fill(elements, intent_label, query=hint, topk=10)
                    elif action_type == 'hover':
                        candidates = retrieve_hover(elements, intent_label, query=hint, topk=10)
                    elif action_type == 'press':
                        candidates = retrieve_press(elements, intent_label, query=hint, topk=10)
                    elif action_type == 'select_option':
                        candidates = retrieve_select_option(elements, intent_label, query=hint, option_query=hint, topk=10)
                    
                    all_scores_zero = all(c.get('score', 0) == 0 for c in candidates) if candidates else True
                    
                    # Ground on the ranked candidates; fall back to the full accessibility tree if none scored
                    if all_scores_zero or not candidates:
                        # Use RAW_GROUNDING
                        logger.info(f"Using RAW_GROUNDING for {action_type} action (all scores zero or no candidates)")
                        user_prompt = RAW_GROUNDING_USER_PROMPT.format(
                            task_goal=self.task,
                            subtask=self.subtask,
                            action_type=action_type,
                            hint=hint,
                            ax_tree=self.current_ax_tree
                        )
                        messages = [
                            {"role": "system", "content": RAW_GROUNDING_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ]
                    else:
                        # Use GROUNDING
                        logger.info(f"Using GROUNDING for {action_type} action")
                        candidate_elements_str = format_candidates_for_llm(candidates, action_type, intent_label)
                        user_prompt = GROUNDING_USER_PROMPT.format(
                            task_goal=self.task,
                            subtask=self.subtask,
                            action_type=action_type,
                            hint=hint,
                            candidate_elements=candidate_elements_str,
                            ax_tree=self.current_ax_tree
                        )
                        messages = [
                            {"role": "system", "content": GROUNDING_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ]
                    
                    # Get grounding response
                    response = litellm.completion(
                        api_key=self.litellm_api_key,
                        base_url=self.litellm_base_url,
                        model=self.model_name,
                        messages=messages,
                        temperature=0.3,
                    )
                    response_text = response.choices[0].message.content
                    action_string = extract_action_from_response(response_text)
                    
                    if not action_string:
                        logger.error(f"Failed to extract action from grounding response: {response_text}")
                        return False, self.env, executed_actions, targeted_element_history, executed_action_strings, ax_tree_history, url_history, self.info_storage, terminated
                        
                # if need text grounding, use TEXT_GROUNDING_PROMPT
                elif action_type == 'send_msg_to_user' or action_type == 'store':
                    # Format the current info storage
                    if self.info_storage:
                        formatted_info = "\n".join([f"- {info}" for info in self.info_storage])
                    else:
                        formatted_info = "No extra information available."
                    logger.info("Using TEXT_GROUNDING for send_msg_to_user or store action")
                    user_prompt = TEXT_GROUNDING_USER_PROMPT.format(
                        task_goal=self.task,
                        subtask=self.subtask,
                        extra_info=formatted_info,
                        ax_tree=self.current_ax_tree
                    )
                    messages = [
                        {"role": "system", "content": TEXT_GROUNDING_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = litellm.completion(
                        api_key=self.litellm_api_key,
                        base_url=self.litellm_base_url,
                        model=self.model_name,
                        messages=messages,
                        temperature=0.3,
                    )
                    response_text = response.choices[0].message.content
                    message = extract_action_from_response(response_text)
                    
                    if not message:
                        logger.error(f"Failed to extract message from text grounding response: {response_text}")
                        return False, self.env, executed_actions, targeted_element_history, executed_action_strings, ax_tree_history, url_history, self.info_storage, terminated
                    
                    # Remove quotes if present
                    message = message.strip("'\"")
                    if action_type == 'send_msg_to_user':
                        action_string = f"send_msg_to_user('{message}')"
                    else:  # Store the extracted information
                        action_string = f"store('{message}')"
                        self.info_storage.append(message)
                
                # if action type is go_back, go_forward, or noop, form action string directly
                elif action_type in {'go_back', 'go_forward', 'noop'}:
                    action_string = f"{action_type}()"
                else:
                    logger.error(f"Unknown action type: {action_type}")
                    return False, self.env, executed_actions, targeted_element_history, executed_action_strings, ax_tree_history, url_history, self.info_storage, terminated
                
                # form action string, execute the action
                if action_type != 'store':
                    # Optional guard: on live pages an action can leave the observation unchanged
                    # (async render, dismissed overlay, non-interactive target), and the planner may
                    # keep re-proposing it. NESY_REPEAT_CAP=N skips an action after N identical tries.
                    repeat_cap = int(os.environ.get("NESY_REPEAT_CAP", 0))
                    if repeat_cap and executed_action_strings.count(action_string) >= repeat_cap:
                        logger.warning(
                            f"Action {action_string} already tried {repeat_cap}x with no progress; "
                            "skipping to let the planner choose another route")
                        executed_action_strings.append(f"# skipped-repeat {action_string}")
                        continue
                    logger.info(f"Executing action: {action_string}")
                    obs, reward, term, trunc, info = self.env.step(action_string)
                    self.recent_obs = obs
                    terminated = term
                    truncated = trunc
                    # Track action string for executed actions
                    executed_action_strings.append(action_string)
                else:
                    # For store action, track the action string even though we don't execute it
                    executed_action_strings.append(action_string)
                    truncated = False
                
                # update all returned types, current ax_tree, and info_storage (if store action is executed)
                # Update current ax_tree

                self.current_ax_tree = flatten_axtree_to_str(self.recent_obs.get("axtree_object", {}))
                
                # get the URL from the observation (only if we executed an action)
                if action_type != 'store':
                    url = self.recent_obs.get("url", "")
                    # Check for unexpected page transition
                    if len(url_history) > 0:
                        last_url = url_history[-1]
                        if url != last_url:
                            logger.info(f"Unexpected page transition detected: {last_url} -> {url}. Breaking action sequence.")
                            url_history.append(url)
                            break
                    url_history.append(url)
                
                # Extract targeted element ID from action string and get full element details
                targeted_element_id = ""
                if action_type in {'click', 'fill', 'hover', 'press', 'select_option'}:
                    targeted_element_id = extract_targeted_element_id_from_action_string(action_string)
                
                # Find the element from the elements list or candidates
                targeted_element = {}
                if targeted_element_id:
                    # First try to find in candidates (if available)
                    if candidates:
                        for candidate in candidates:
                            if str(candidate.get('id')) == str(targeted_element_id):
                                targeted_element = {
                                    "id": candidate.get('id'),
                                    "role": candidate.get('role'),
                                    "value": candidate.get('value'),
                                    "attributes": candidate.get('attributes'),
                                    "context": candidate.get('context')
                                }
                                break
                    
                    # If not found in candidates, search in all elements
                    if not targeted_element and elements:
                        for element in elements:
                            if str(element.get('id')) == str(targeted_element_id):
                                targeted_element = {
                                    "id": element.get('id'),
                                    "role": element.get('role'),
                                    "value": element.get('value'),
                                    "attributes": element.get('attributes'),
                                    "context": element.get('context')
                                }
                                break
                    
                    # If still not found, create minimal dict with just the ID
                    if not targeted_element:
                        targeted_element = {
                            "id": targeted_element_id,
                            "role": "",
                            "value": "",
                            "attributes": [],
                            "context": {}
                        }
                else:
                    # No element ID (for actions like go_back, go_forward, noop, send_msg_to_user)
                    targeted_element = {
                        "id": "",
                        "role": "",
                        "value": "",
                        "attributes": [],
                        "context": {}
                    }
                
                targeted_element_history.append(targeted_element)
                
                # Update executed_actions with action object
                executed_actions.append({
                    "action": action_type,
                    "intent": intent_label,
                    "hint": hint
                })
                ax_tree_history.append(self.current_ax_tree)
                
                # if the environment was terminated, break
                if action_type != 'store' and terminated:
                    logger.info("Environment was terminated")
                    break
                
                # if the environment was truncated, break
                if action_type != 'store' and truncated:
                    logger.warning("Environment was truncated")
                    break
                
                # current_action_length + 1
                current_action_length += 1
            
            # use judge model to judge if the subtask is completed, if yes, set completed to True
            logger.info("Judging if subtask is completed")
            judge_user_prompt = JUDGE_USER_PROMPT.format(
                executed_subgoal=self.subtask,
                ax_tree=self.current_ax_tree,
                action_history=executed_action_strings
            )
            messages = [
                {"role": "system", "content": SUBGOAL_JUDGE_PROMPT},
                {"role": "user", "content": judge_user_prompt}
            ]
            
            response = litellm.completion(
                api_key=self.litellm_api_key,
                base_url=self.litellm_base_url,
                model=self.model_name,
                messages=messages,
                temperature=0.3,
            )
            response_text = response.choices[0].message.content
            
            judge_log = os.environ.get("NESY_JUDGE_LOG")
            if judge_log:
                try:
                    with open(judge_log, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps({
                            "subgoal": self.subtask,
                            "actions": list(executed_action_strings),
                            "response": response_text}, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            # Extract completed or not_completed from the response
            if "<completed>" in response_text:
                completed = True
                logger.info("Judge determined subtask is completed")
            elif "<failed>" in response_text:
                completed = False
                logger.info("Judge determined subtask is not completed")
                break
            else:
                # Fallback: try to extract from response
                judge_result = extract_action_from_response(response_text)
                if judge_result:
                    completed = judge_result.strip().lower() in ["completed", "success", "true"]
                elif "<incomplete>" in response_text:
                    completed = False
                    logger.info("Judge determined subtask is incomplete")
                else:
                    logger.warning(f"Could not determine completion status from judge response: {response_text}")
                    completed = False
        
        # return as specified in the signature
        return completed, self.env, executed_actions, targeted_element_history, executed_action_strings, ax_tree_history, url_history, self.info_storage, terminated