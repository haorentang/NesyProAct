"""
Evaluator Module - Evaluates task execution success using text-based evaluation
"""
import os
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import litellm
from autoeval.prompts import build_text_eval_prompt
from browsergym.utils.obs import flatten_axtree_to_str

logger = logging.getLogger(__name__)


def extract_evaluation_result(response_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract thoughts and status from LLM evaluation response

    Args:
        response_text: LLM response text

    Returns:
        Tuple of (thoughts: Optional[str], status: Optional[str])
    """
    thoughts = None
    status = None

    # Try to extract Thoughts: line
    thoughts_match = re.search(r'Thoughts:\s*(.+?)(?=\n|$)', response_text, re.DOTALL | re.IGNORECASE)
    if thoughts_match:
        thoughts = thoughts_match.group(1).strip()

    # Try to extract Status: line
    status_match = re.search(r'Status:\s*["\']?(\w+)["\']?', response_text, re.IGNORECASE)
    if status_match:
        status = status_match.group(1).lower()

    return thoughts, status


def evaluate_task_execution(
    intent: str,
    action_history: List[Dict[str, Any]],
    response: Optional[str] = None,
    final_obs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate task execution using text-based evaluation

    Args:
        intent: User's intent/task description
        action_history: List of actions executed during task
        response: Bot's response to the user (optional)
        final_obs: Final observation from environment (optional, for axtree extraction)

    Returns:
        Dictionary with evaluation results:
        {
            "success": bool,
            "thoughts": str,
            "status": str ("success" or "failure")
        }
    """
    try:
        # Format action history
        action_history_str = "\n".join([
            f"{idx + 1}. {action.get('action', 'unknown')} - {action.get('description', '')}"
            for idx, action in enumerate(action_history)
        ])

        # Extract axtree from final_obs if available
        cap = ""
        if final_obs and "axtree_object" in final_obs:
            try:
                cap = flatten_axtree_to_str(final_obs["axtree_object"])
            except Exception as e:
                logger.warning(f"Failed to flatten axtree for evaluation: {e}")
                cap = ""

        # Build prompt using text-based evaluation
        prompt, system_msg = build_text_eval_prompt(
            cap=cap,
            intent=intent,
            response=response or "N/A",
            last_actions=action_history_str
        )

        # Call LLM for evaluation
        response_obj = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL"),
            model=os.environ.get("NESY_EVAL_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        response_text = response_obj.choices[0].message.content.strip()

        # Extract evaluation result
        thoughts, status = extract_evaluation_result(response_text)

        # Determine success
        success = status == "success" if status else False

        result = {
            "success": success,
            "thoughts": thoughts or response_text,
            "status": status or "unknown",
            "raw_response": response_text
        }

        logger.info(f"Evaluation result: {status} - {thoughts[:100] if thoughts else 'N/A'}")

        return result

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "thoughts": f"Evaluation error: {str(e)}",
            "status": "error",
            "raw_response": None
        }


def collect_action_history_from_memory(memory_entries: List[Any]) -> List[Dict[str, Any]]:
    """
    Collect action history from memory entries

    Args:
        memory_entries: List of MemoryEntry objects

    Returns:
        List of action dictionaries with 'action' and 'description' keys
    """
    action_history = []

    for entry in memory_entries:
        if entry.steps:
            for step in entry.steps:
                action_history.append({
                    "action": step.action_type,
                    "intent": step.action_reason,
                    "step_index": step.step_index,
                })

    return action_history


def collect_response_from_memory(memory_entries: List[Any]) -> Optional[str]:
    """
    Collect bot response from memory entries (look for send_msg_to_user actions)

    Args:
        memory_entries: List of MemoryEntry objects

    Returns:
        Combined bot response string containing all send_msg_to_user messages, or None if none found
    """
    responses = []
    for entry in memory_entries:
        if entry.steps:
            for step in entry.steps:
                if step.action_type == "send_msg_to_user":
                    msg = step.action_str
                    if msg:
                        responses.append(msg)

    if responses:
        return "\n".join(responses)
    return None
