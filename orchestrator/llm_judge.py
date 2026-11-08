"""
LLM Judge Module

Provides LLM-based evaluation and skill generation for subtask execution.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import litellm
from browsergym.utils.obs import flatten_axtree_to_str

from memory.memory import ExecutionStep
from orchestrator.prompts import LLM_JUDGE_SUBTASK_PROMPT, LLM_JUDGE_SUBTASK_USER_PROMPT

logger = logging.getLogger(__name__)


def llm_judge_subtask(
    query: str,
    steps: List[ExecutionStep],
    page_state: Dict[str, Any],
    obs: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Use LLM to evaluate subtask execution and generate skill representation
    
    This should be called for all subtasks regardless of execution path.
    Generates a generalized skill name, description, and parameters.
    
    Args:
        query: Subtask query
        steps: List of execution steps that were performed
        page_state: Current page state after execution
        obs: Optional browsergym observation dictionary (from browsergym_executor.get_browsergym_obs())
        model_name: LLM model name (if None, reads from LLM_MODEL_NAME env var, defaults to "gpt-4o-mini")
    
    Returns:
        Tuple of (skill_name: Optional[str], skill_description: Optional[str], skill_parameter: Optional[List[str]])
        Returns skill representation as generalized versions (e.g., "search_item" instead of "search for banana", ["item_name"] as parameter)
    """
    # Get model_name from environment variable if not provided
    if model_name is None:
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    try:
        # Prepare steps summary
        steps_summary = []
        for step in steps:
            steps_summary.append({
                "action": step.action_str,
                "type": step.action_type
            })
        
        url_after = obs.get("url", "") if obs else page_state.get("url", "")
        error = obs.get("last_action_error", "") if obs else ""
        
        # Extract AXTree if available
        axtree_txt = ""
        if obs and "axtree_object" in obs:
            try:
                axtree_txt = flatten_axtree_to_str(obs["axtree_object"])
            except Exception as e:
                logger.warning(f"Failed to flatten AXTree: {e}")
                axtree_txt = ""
        
        # Determine execution success from steps (no error and steps were executed)
        execution_success = len(steps) > 0 and not error
        
        # Prepare AXTree section
        ax_tree_section = ""
        if axtree_txt:
            ax_tree_section = f"# Current page Accessibility Tree\n\n{axtree_txt}"
        else:
            ax_tree_section = "# Current page Accessibility Tree\n\n(Not available)"
        
        # Build user prompt using template
        user_prompt = LLM_JUDGE_SUBTASK_USER_PROMPT.format(
            query=query,
            num_steps=len(steps),
            steps_summary=json.dumps(steps_summary, indent=2),
            url_after=url_after,
            error=error if error else "None",
            execution_success=execution_success,
            ax_tree_section=ax_tree_section
        )

        response = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_BASE_URL"),
            model=model_name,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SUBTASK_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Try direct parse; on failure, extract first JSON object
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                logger.warning(f"Could not extract JSON from llm_judge response: {content[:200]}")
                result = {}
        skill_name = result.get("skill_name")
        skill_description = result.get("skill_description")
        logger.info(f"LLM judge parsed result: skill_name={skill_name}, skill_description={skill_description}, raw_keys={list(result.keys())}")

        # If LLM didn't return skill_name, don't save
        if not skill_name:
            return None, None, None

        has_fill_action = any(step.action_type == "fill" for step in steps)
        skill_parameter = ["parameter"] if has_fill_action else []
        logger.info(f"has_fill_action={has_fill_action}, skill_name={skill_name}, skill_parameter={skill_parameter}")

        return skill_name, skill_description, skill_parameter
        
    except Exception as e:
        logger.warning(f"LLM judge subtask failed: {e}")
        return None, None, None


