import re
import yaml
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
from browsergym.utils.obs import flatten_axtree_to_str
from datetime import datetime
from memory.memory import Memory, MemoryEntry
import json
from pathlib import Path
from interface.utils import extract_all_elements
from urllib.parse import urlparse
import litellm

logger = logging.getLogger(__name__)


def extract_and_parse_yaml_action_sequence(response: str) -> list[dict]:
    """
    Extract and parse the YAML action sequence from the model response.
    Args:
        response: The model response containing the YAML action sequence.
    Returns:
        A list of action objects in the format:
        {
            "action": <action_type>,
            "intent": <intent_label>,
            "hint": <action_hint>
        }
    """
    # Try to extract YAML from markdown code blocks
    yaml_match = re.search(r'```yaml\s*\n(.*?)```', response, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
    else:
        # Try to extract YAML from generic code blocks
        yaml_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
        else:
            # Try to find YAML content directly (lines starting with -)
            lines = response.split('\n')
            yaml_lines = []
            in_yaml = False
            for line in lines:
                if line.strip().startswith('- action:'):
                    in_yaml = True
                if in_yaml:
                    yaml_lines.append(line)
                    # Stop if we hit a line that looks like end of YAML and is not indented
                    if line.strip() and not line.startswith((' ', '-', '\t')) and not line.strip().startswith('#'):
                        if not any(keyword in line for keyword in ['action:', 'intent:', 'hint:']):
                            break
            yaml_content = '\n'.join(yaml_lines).strip()

    if not yaml_content:
        return []

    try:
        # Parse YAML
        parsed = yaml.safe_load(yaml_content)

        # Handle case where parsed is a list
        if isinstance(parsed, list):
            return parsed

        # Handle case where parsed is a dict with a list inside
        if isinstance(parsed, dict):
            # Look for common keys that might contain the list
            for key in ['actions', 'sequence', 'workflow']:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # If it's a single action dict, wrap it in a list
            if 'action' in parsed:
                return [parsed]

        return []
    except yaml.YAMLError as e:
        # If YAML parsing fails, return empty list
        return []

def validate_yaml_action_sequence(action_sequence: list[dict]) -> tuple[bool, str]:
    """
    Validate the YAML action sequence:
    - The YAML should be in the correct format and structure.
    - Check if all actions are paired with an intent label and a hint.
    Args:
        action_sequence: The YAML action sequence to validate.
    Returns:
        A tuple containing a boolean indicating whether the action sequence is valid and a string explaining the validation failure if it is not valid.
    """
    if not isinstance(action_sequence, list):
        return False, "Action sequence must be a list"

    if len(action_sequence) == 0:
        return False, "Action sequence cannot be empty"

    # Valid action types
    valid_action_types = {'click', 'fill', 'hover', 'press', 'go_back', 'go_forward', 'send_msg_to_user', 'noop', 'store', 'select_option'}

    # Valid intent labels (dynamically imported from INTENT_LABEL)
    from interface.intents import INTENT_LABEL
    valid_intent_labels = set(INTENT_LABEL.keys())

    for i, action in enumerate(action_sequence):
        if not isinstance(action, dict):
            return False, f"Action at index {i} must be a dictionary"

        # Check for required 'action' field
        if 'action' not in action:
            return False, f"Action at index {i} is missing the 'action' field"

        action_type = action['action']

        # Validate action type
        if action_type not in valid_action_types:
            return False, f"Action at index {i} has invalid action type '{action_type}'. Valid types are: {', '.join(sorted(valid_action_types))}"

        # Check for required 'intent' field
        if 'intent' not in action:
            return False, f"Action at index {i} is missing the 'intent' field"

        intent_label = action['intent']

        # Validate intent label
        if not isinstance(intent_label, str):
            return False, f"Action at index {i} has 'intent' field that is not a string"

        if intent_label not in valid_intent_labels:
            return False, f"Action at index {i} has invalid intent label '{intent_label}'. Valid labels are: {', '.join(sorted(valid_intent_labels))}"

        # Check for required 'hint' field
        if 'hint' not in action:
            return False, f"Action at index {i} is missing the 'hint' field"

        hint = action['hint']

        # Validate hint
        if not isinstance(hint, str):
            return False, f"Action at index {i} has 'hint' field that is not a string"

        if not hint.strip():
            return False, f"Action at index {i} has 'hint' field that is empty or contains only whitespace"

    return True, ""

def extract_action_from_response(response: str) -> str:
    """
    Extract the action from the model response.
    Args:
        response: The model response containing the action.
    Returns:
        The action calling format like click(bid='<bid>') or fill(bid='<bid>', text='<text>') or hover(bid='<bid>') or press(bid='<bid>', 'Enter') or send_msg_to_user(text='<text>').
    """
    # Extract content between angle brackets <...>
    # Pattern matches: <action_function_string> or <'message'>
    match = re.search(r'<([^>]+)>', response)

    if match:
        action_string = match.group(1).strip()
        return action_string

    # If no match found, return empty string
    return ""

def extract_targeted_element_id_from_action_string(action_string: str) -> str:
    """
    Extract targeted element id (bid) from action string.

    Supports:
      click("64")
      fill("269", "camera")
      select_option("135", "Hot")
      select_option(bid="135", options="Hot")
      hover('64')

    Returns:
      element id as string, or "" if not found.
    """
    if not action_string:
        return ""

    s = action_string.strip()

    # 1) Try keyword argument: bid="123"
    m = re.search(r"\bbid\s*=\s*['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)

    # 2) Try first positional argument: action("123", ...)
    m = re.search(r"\(\s*['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)

    return ""



def get_instance_path(skill_name: str, instance_name: str, mem_dir: str = "mem") -> tuple[str, str]:
    """
    Get paths for execution and axtree directories for a given instance

    Args:
        skill_name: Skill name
        instance_name: Instance name
        mem_dir: Memory directory path (default: "mem")

    Returns:
        Tuple of (execution_dir, axtree_dir)
    """
    base_dir = os.path.join(mem_dir, skill_name, instance_name)
    execution_dir = os.path.join(base_dir, "execution")
    axtree_dir = os.path.join(base_dir, "axtree")
    return execution_dir, axtree_dir


def save_axtree_str(axtree_str: str, instance_name: str, skill_name: str, step_index: int, suffix: str = "", mem_dir: str = "mem"):
    """
    Save axtree string directly to file system: {mem_dir}/skill_name/instance_name/axtree/step_index.txt

    Args:
        axtree_str: AXTree string to save
        instance_name: Instance name
        skill_name: Skill name
        step_index: Step index
        suffix: Optional suffix for filename (e.g., "_post" for post-action axtree)
    """
    try:
        _, axtree_dir = get_instance_path(skill_name, instance_name, mem_dir)
        os.makedirs(axtree_dir, exist_ok=True)

        # Generate filename: step_index.txt or step_index_suffix.txt
        if suffix:
            filename = f"{step_index:03d}{suffix}.txt"
        else:
            filename = f"{step_index:03d}.txt"  # Use 3-digit zero-padded format
        filepath = os.path.join(axtree_dir, filename)

        # Save axtree string directly to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(axtree_str)

        logger.debug(f"Saved axtree string for step {step_index} to: {filepath}")

    except Exception as e:
        logger.warning(f"Failed to save axtree string for step {step_index}: {e}")

def generate_instance_name() -> str:
        """
        Generate instance name using timestamp

        Returns:
            Instance name string (e.g., "20260116_172053_410230")
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")








def save_execution_experience(
        skill_name: str,
        memory_entry: MemoryEntry,
        success: bool,
        mem_dir: str = "mem"
    ) -> Optional[str]:
        """
        Save execution experience to file system: {mem_dir}/skill_name/instance_name/execution

        Args:
            skill_name: Skill name
            memory_entry: Memory entry to save
            success: Whether execution succeeded
            mem_dir: Memory directory path (default: "mem")

        Returns:
            Instance name that was used
        """
        try:

            instance_name = generate_instance_name()

            # Create directory structure: {mem_dir}/skill_name/instance_name/execution
            execution_dir, axtree_dir = get_instance_path(skill_name, instance_name, mem_dir)
            os.makedirs(execution_dir, exist_ok=True)
            os.makedirs(axtree_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            status = "success" if success else "failure"
            filename = f"{timestamp}_{status}.json"
            filepath = os.path.join(execution_dir, filename)

            # Prepare data to save (use memory_entry.to_dict() to avoid duplication)
            execution_data = memory_entry.to_dict()
            execution_data["timestamp"] = datetime.now().isoformat()
            execution_data["success"] = success
            execution_data["instance_name"] = instance_name  # Store instance name in execution data

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=2, ensure_ascii=False)

            # Save axtree for each step to file system
            for step in memory_entry.steps:
                if step.axtree:
                    save_axtree_str(step.axtree, instance_name, skill_name, step.step_index, mem_dir=mem_dir)

            logger.info(f"Saved execution experience to: {filepath}")
            logger.info(f"Instance path: {os.path.join(mem_dir, skill_name, instance_name)}")

            return instance_name

        except Exception as e:
            logger.error(f"Failed to save execution experience: {e}")
            return None

def load_skill_from_mem(skill_name: str, mem_dir: str = "mem") -> Optional[Dict[str, Any]]:
        """
        Load a skill from mem directory by skill name

        Directory structure: mem/skill_name/instance_name/execution/*_success.json

        Args:
            skill_name: Name of the skill to load
            mem_dir: Memory directory path (default: "mem")

        Returns:
            Dictionary with skill data if found, None otherwise
        """
        if not skill_name:
            return None

        skill_dir = Path(mem_dir) / skill_name

        if not skill_dir.exists():
            logger.warning(f"Skill directory {skill_dir} does not exist")
            return None

        # Find the first instance directory under this skill
        instance_dirs = [d for d in skill_dir.iterdir() if d.is_dir()]
        if not instance_dirs:
            logger.warning(f"No instance directories found for skill {skill_name}")
            return None

        # Use the first instance directory (sorted by name for consistency)
        first_instance_dir = sorted(instance_dirs, key=lambda p: p.name)[0]
        execution_dir = first_instance_dir / "execution"

        if not execution_dir.exists():
            logger.warning(f"Execution directory {execution_dir} does not exist")
            return None

        # Find the most recent success JSON file
        success_files = list(execution_dir.glob("*_success.json"))
        if not success_files:
            logger.warning(f"No success files found for skill {skill_name}")
            return None

        # Sort by modification time, get the most recent
        most_recent = max(success_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(most_recent, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Failed to load skill {skill_name} from {most_recent}: {e}")
            return None


def find_element_bid_by_value(
    axtree: str,
    target_value: str,
    prev_step: Dict[str, Any],
    action_type: str
) -> Optional[str]:
    """
    Find the element in the current page's accessibility tree whose value matches the stored target value.

    Strategy:
    1. Match an element by its value.
    2. Otherwise fall back to the stored step's original bid if that bid still exists on the page.

    Args:
        axtree: Flattened accessibility tree of the current page
        target_value: Value attribute recorded for the target element
        prev_step: Stored execution step (contains original_bid, etc.)
        action_type: Action type ('click', 'fill', 'hover', 'press', 'select_option')

    Returns:
        The matched bid, or None if no element was found.
    """
    if not target_value or action_type not in {'click', 'fill', 'hover', 'press', 'select_option'}:
        return None

    # Extract all elements from axtree
    elements = extract_all_elements(axtree)

    matched_bid = None
    find_bid = False
    original_bid = prev_step.get("original_bid")

    # Search for element with matching value
    for element in elements:
        element_value = element.get("value", "")
        if element_value == target_value:
            matched_bid = element.get("id")
            if matched_bid:
                logger.info(f"Found matching element: value='{target_value}', bid='{matched_bid}'")
                break

        # Check if this element has the same bid as previous step
        bid = element.get("id")
        if bid and original_bid and bid == original_bid:
            find_bid = True

    # If no match found by value, try to use original_bid (might work if page hasn't changed)
    if not matched_bid:
        if find_bid and original_bid:
            matched_bid = original_bid
            logger.info(f"Using original_bid as fallback: '{original_bid}'")

    return matched_bid

def urls_match_with_one_section_diff(url1: str, url2: str) -> bool:
    """
    Check if two URLs match with at most one section difference.

    URLs are split into sections (path segments). Two URLs match if:
    - They have the same number of sections
    - At most one section differs at the same position

    Examples:
    - "http://example.com/products/item1" and "http://example.com/products/item2" -> True (non-admin)
      (sections: ['products', 'item1'] vs ['products', 'item2'], one difference)
    - "http://example.com/products/item1" and "http://example.com/categories/item1" -> True (non-admin)
      (sections: ['products', 'item1'] vs ['categories', 'item1'], one difference)
    - "http://example.com/products/item1" and "http://example.com/products/item2/details" -> False
      (different number of sections: 2 vs 3)
    - "http://example.com/products/item1" and "http://example.com/categories/item2" -> False
      (two differences: 'products' vs 'categories' and 'item1' vs 'item2')

    Args:
        url1: First URL string
        url2: Second URL string

    Returns:
        True if URLs match, False otherwise
    """
    if not url1 or not url2:
        return False

    try:
        # Parse URLs
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)

        # Extract path sections (remove leading/trailing slashes and split)
        path1 = parsed1.path.strip('/')
        path2 = parsed2.path.strip('/')

        # Split into sections
        sections1 = [s for s in path1.split('/') if s] if path1 else []
        sections2 = [s for s in path2.split('/') if s] if path2 else []

        # Must have same number of sections
        if len(sections1) != len(sections2):
            return False

        # If both are empty (root path), they match
        if len(sections1) == 0:
            return True

        # Count differences
        differences = 0
        for i in range(len(sections1)):
            if sections1[i] != sections2[i]:
                differences += 1
                if differences > 1:
                    return False

        # At most one difference is allowed
        return True

    except Exception as e:
        logger.warning(f"Error comparing URLs {url1} and {url2}: {e}")
        return False

def get_available_skills_with_url_match(mem_dir: str, current_url: str) -> List[Dict[str, Any]]:
    """
    Get available skills that match the current URL

    Traverses all skills and all their instances, checking url_before for each instance.
    Returns all skills that have at least one instance matching the current URL.

    Args:
        mem_dir: Skill-memory directory for the current site and backbone
        current_url: Current page URL

    Returns:
        List of skill dictionaries with 'name', 'description', and 'parameters' keys
        Each skill appears only once, even if multiple instances match
    """
    skills = []
    mem_path = Path(mem_dir)

    if not mem_path.exists():
        logger.debug(f"Memory directory {mem_dir} does not exist")
        return skills

    # Track skills we've already added (to avoid duplicates)
    added_skill_names = set()

    # Iterate through all skill directories
    for skill_dir in mem_path.iterdir():
        if not skill_dir.is_dir():
            continue

        skill_name = skill_dir.name

        # Find all instance directories under this skill
        instance_dirs = [d for d in skill_dir.iterdir() if d.is_dir()]
        if not instance_dirs:
            continue

        # Iterate through all instances (not just the first one)
        for instance_dir in instance_dirs:
            execution_dir = instance_dir / "execution"

            if not execution_dir.exists():
                continue

            # Find all success JSON files in this instance
            success_files = list(execution_dir.glob("*_success.json"))
            if not success_files:
                continue

            # Get the most recent file from this instance
            most_recent = max(success_files, key=lambda p: p.stat().st_mtime)

            try:
                with open(most_recent, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    skill_url_before = data.get("url_before", "")

                    # Check URL match
                    if skill_url_before and urls_match_with_one_section_diff(skill_url_before, current_url):
                        # URL matches, add this skill if not already added
                        skill_name_from_data = data.get("skill_name", skill_name)

                        # Skip if we've already added this skill
                        if skill_name_from_data in added_skill_names:
                            continue

                        skill_description = data.get("skill_description", "")
                        skill_parameter = data.get("skill_parameter", [])

                        if skill_name_from_data:
                            skills.append({
                                "name": skill_name_from_data,
                                "description": skill_description,
                                "parameters": skill_parameter if isinstance(skill_parameter, list) else []
                            })
                            added_skill_names.add(skill_name_from_data)
            except Exception as e:
                logger.warning(f"Failed to read skill file {most_recent}: {e}")
                continue

    logger.debug(f"Found {len(skills)} available skills matching URL")
    return skills


def decide_skill_reuse(
    task_goal: str,
    subtask_goal: str,
    available_skills: List[Dict[str, Any]],
    model_name: Optional[str] = None
) -> Optional[Tuple[str, Optional[str]]]:
    """
    Use reuse agent to decide if any skill can be reused

    Args:
        task_goal: Overall task goal
        subtask_goal: Current subtask goal/query
        available_skills: List of available skills with name, description, and parameters
        model_name: Model name for LLM calls
        openai_client: OpenAI client instance (if None, will create one)

    Returns:
        Tuple of (skill_name, parameter) if a skill can be reused, None otherwise
        parameter can be None if the skill doesn't require parameters
    """
    from orchestrator.prompts import SKILL_REUSE_PROMPT, SKILL_REUSE_USER_PROMPT

    if not available_skills:
        return None

    # Get model_name from environment variable if not provided
    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    # Format available skills for prompt (include parameter information)
    skills_list = []
    for skill in available_skills:
        skill_str = f"- Skill name: {skill['name']}"
        if skill.get('description'):
            skill_str += f"Skill description: {skill['description']}"
        parameters = skill.get('parameters', [])
        if parameters:
            skill_str += f" (parameter required: True)"
        else:
            skill_str += f" (parameter required: False)"
        skills_list.append(skill_str)

    skills_available_str = "\n".join(skills_list)

    # Build prompt
    user_prompt = SKILL_REUSE_USER_PROMPT.format(
        task_goal=task_goal,
        subtask_goal=subtask_goal,
        skills_avaliable=skills_available_str
    )


    try:
        response = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL"),
            model=model_name,
            messages=[
                {"role": "system", "content": SKILL_REUSE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )



        content = response.choices[0].message.content.strip()

        # Extract JSON: try code blocks first, then find outermost {}
        json_str = re.search(r'```(?:json)?\s*\n(.*?)```', content, re.DOTALL)
        json_str = json_str.group(1).strip() if json_str else None
        if not json_str:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if not json_str:
            logger.warning(f"No JSON found in response: {content[:200]}")
            return None

        # Parse and extract name/parameter
        data = json.loads(json_str)


        name = data.get("name")
        param = data.get("parameter")
        if not name:
            return None
        return (name, None if param is None or str(param).lower() == 'null' else param)

    except Exception as e:
        logger.error(f"Failed to decide skill reuse: {e}")
        return None