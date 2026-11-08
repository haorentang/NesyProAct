import dataclasses
import json
import re
import logging
from enum import Enum
from typing import List, Optional, Dict, Any
import os
import litellm
from planner.prompt import SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not content or not content.strip():
        return {}
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning(f"Failed to parse JSON from response: {content[:200]}")
    return {}


class TaskStatus(Enum):
    """Subtask Status"""
    PENDING = "pending"  # Pending
    IN_PROGRESS = "in_progress"  # In Progress
    COMPLETED = "completed"  # Completed
    FAILED = "failed"  # Failed
    SKIPPED = "skipped"  # Skipped


@dataclasses.dataclass
class SubTask:
    """Subtask Data Structure"""
    id: int  # Subtask ID
    description: str  # Subtask Description
    status: TaskStatus = TaskStatus.PENDING  # Task Status
    dependencies: List[int] = dataclasses.field(default_factory=list)  # Dependencies of other subtasks
    result: Optional[str] = None  # Execution Result
    error: Optional[str] = None  # Error Information
    precondition_predicates: Optional[Dict[str, Any]] = None  # Precondition predicates for proactive grounding
    # Format: {"previous_actions": [action_types], "url": [keywords], "axtree": [keywords]}
    # "previous_actions": action types that must appear in previous action sequences
    # "url": keywords that must appear in the current URL
    # "axtree": keywords that must appear in the current page accessibility tree


class Planner:
    """
    Task Planner - Decompose a complex task into multiple subtasks
    
    Use LLM to decompose the user's task goal into a series of ordered subtasks,
    and track the execution status of each subtask.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_subtasks: int = 10,
        enable_parallel: bool = False,
        site: str = "default"
    ):
        """
        Initialize the planner
        
        Args:
            model_name: The name of the LLM model for task decomposition
            max_subtasks: The maximum number of subtasks
            enable_parallel: Whether to allow parallel execution (considering dependencies)
            site: Site name of the task
        """
        self.model_name = model_name
        self.max_subtasks = max_subtasks
        self.enable_parallel = enable_parallel
        
        self.original_goal: Optional[List[Dict[str, Any]]] = None
        self.subtasks: List[SubTask] = []
        self.current_subtask_index: int = 0
        self.site = site
        
    def plan(
        self, 
        goal_object: tuple[Dict[str, Any]],
        axtree: Optional[str] = None,
        url: Optional[str] = None,
        env: Optional[Any] = None
    ) -> List[SubTask]:
        """
        Decompose the task goal into subtasks
        
        Args:
            goal_object: The task goal, a list of messages in OpenAI style
            axtree: Current page accessibility tree as string (optional)
            url: Current page URL (optional)
            env: Browser environment (unused by the planner; kept for API compatibility)
            
        Returns:
            The list of subtasks
        """
        self.original_goal = goal_object
        
        # Extract the task description text
        goal_text = self._extract_goal_text(goal_object)
        
        obs = None
        # Decompose the task using the LLM
        subtasks = self._decompose_task(goal_text, axtree=axtree, url=url)
        
        # Validate and limit the number of subtasks
        if len(subtasks) > self.max_subtasks:
            logger.warning(
                f"The number of subtasks ({len(subtasks)}) exceeds the maximum limit ({self.max_subtasks}),"
                f"will take the first {self.max_subtasks} subtasks"
            )
            subtasks = subtasks[:self.max_subtasks]
        
        self.subtasks = subtasks
        self.current_subtask_index = 0
        
        logger.info(f"The task has been decomposed into {len(self.subtasks)} subtasks")
        for i, subtask in enumerate(self.subtasks, 1):
            logger.info(f"  Subtask {i}: {subtask.description}")

        return self.subtasks, obs
    
    def _extract_goal_text(self, goal_object: tuple[Dict[str, Any]]) -> str:
        """Extract the text description from the goal object"""
        texts = []
        for msg in goal_object:
            if msg.get("type") == "text":
                texts.append(msg.get("text", ""))
        return "\n".join(texts)
    
    def _decompose_task(
        self, 
        goal_text: str,
        axtree: Optional[str] = None,
        url: Optional[str] = None
    ) -> List[SubTask]:
        """
        Decompose the task using LLM
        
        Args:
            goal_text: The text description of the task
            axtree: Current page accessibility tree as string (optional)
            url: Current page URL (optional)
            
        Returns:
            The list of subtasks
        """
        # Build user prompt with task, axtree, and url
        user_prompt = USER_PROMPT.format(
            task=goal_text,
            url=url if url else "N/A",
        )

        try:
            response = litellm.completion(
                api_key=os.environ.get("LITELLM_API_KEY"),
                base_url=os.environ.get("LITELLM_BASE_URL"),
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            content = response.choices[0].message.content
            result = _parse_json_response(content)

            subtasks = []
            for item in result.get("subtasks", []):
                subtask = SubTask(
                    id=item.get("id", len(subtasks) + 1),
                    description=item.get("description", ""),
                    dependencies=item.get("dependencies", []),
                    precondition_predicates=item.get("precondition_predicates")  # Precondition predicates
                )
                subtasks.append(subtask)
            
            # # Validate the dependencies
            # self._validate_dependencies(subtasks)
            
            return subtasks
            
        except Exception as e:
            raise e
    
    def _validate_dependencies(self, subtasks: List[SubTask]):
        """Validate the dependencies of the subtasks"""
        subtask_ids = {st.id for st in subtasks}
        
        for subtask in subtasks:
            for dep_id in subtask.dependencies:
                if dep_id not in subtask_ids:
                    logger.warning(
                        f"The dependency {dep_id} of subtask {subtask.id} does not exist, will ignore it"
                    )
                    subtask.dependencies.remove(dep_id)
                if dep_id >= subtask.id:
                    logger.warning(
                        f"The dependency {dep_id} of subtask {subtask.id} is invalid (the dependency should be before the current task), will ignore it"
                    )
                    subtask.dependencies.remove(dep_id)
    
    def get_current_subtask(self) -> Optional[SubTask]:
        """Get the current subtask to execute"""
        if not self.subtasks:
            return None
        
        # Find the next subtask to execute
        for i in range(self.current_subtask_index, len(self.subtasks)):
            subtask = self.subtasks[i]
            
            # Check if all dependencies are completed
            if self._check_dependencies_completed(subtask):
                if subtask.status == TaskStatus.PENDING:
                    subtask.status = TaskStatus.IN_PROGRESS
                    self.current_subtask_index = i
                    return subtask
        
        return None
    
    def _check_dependencies_completed(self, subtask: SubTask) -> bool:
        """Check if all dependencies of the subtask are completed"""
        if not subtask.dependencies:
            return True
        
        for dep_id in subtask.dependencies:
            dep_subtask = self.get_subtask_by_id(dep_id)
            if dep_subtask is None or dep_subtask.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def get_subtask_by_id(self, task_id: int) -> Optional[SubTask]:
        """Get the subtask by ID"""
        for subtask in self.subtasks:
            if subtask.id == task_id:
                return subtask
        return None
    
    def mark_subtask_completed(self, task_id: int, result: Optional[str] = None):
        """Mark the subtask as completed"""
        subtask = self.get_subtask_by_id(task_id)
        if subtask:
            subtask.status = TaskStatus.COMPLETED
            subtask.result = result
            logger.info(f"Subtask {task_id} has been completed: {subtask.description}")
    
    def mark_subtask_failed(self, task_id: int, error: Optional[str] = None):
        """Mark the subtask as failed"""
        subtask = self.get_subtask_by_id(task_id)
        if subtask:
            subtask.status = TaskStatus.FAILED
            subtask.error = error
            logger.warning(f"Subtask {task_id} failed: {subtask.description}, error: {error}")
    
    def mark_subtask_skipped(self, task_id: int, reason: Optional[str] = None):
        """Mark the subtask as skipped"""
        subtask = self.get_subtask_by_id(task_id)
        if subtask:
            subtask.status = TaskStatus.SKIPPED
            subtask.result = reason
            logger.info(f"Subtask {task_id} has been skipped: {subtask.description}, reason: {reason}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get the progress of the task execution"""
        if not self.subtasks:
            return {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "failed": 0,
                "pending": 0,
                "progress_percentage": 0.0,
            }
        
        total = len(self.subtasks)
        completed = sum(1 for st in self.subtasks if st.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for st in self.subtasks if st.status == TaskStatus.IN_PROGRESS)
        failed = sum(1 for st in self.subtasks if st.status == TaskStatus.FAILED)
        pending = sum(1 for st in self.subtasks if st.status == TaskStatus.PENDING)
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "pending": pending,
            "progress_percentage": (completed / total * 100) if total > 0 else 0.0,
        }
    
    def is_all_completed(self) -> bool:
        """Check if all subtasks are completed"""
        if not self.subtasks:
            return False
        
        for subtask in self.subtasks:
            if subtask.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED):
                return False
        
        return True
    
    def get_remaining_subtasks(self) -> List[SubTask]:
        """Get all remaining subtasks"""
        return [
            st for st in self.subtasks
            if st.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        ]
    
    def get_subtask_summary(self) -> str:
        """Get the summary of the subtasks (for passing to the agent)"""
        if not self.subtasks:
            return ""
        
        summary_lines = ["# Task Decomposition"]
        summary_lines.append(f"Total tasks: {len(self.subtasks)}")
        summary_lines.append("")
        
        progress = self.get_progress()
        summary_lines.append(f"Progress: {progress['completed']}/{progress['total']} Completed ({progress['progress_percentage']:.1f}%)")
        summary_lines.append("")
        
        summary_lines.append("## Subtask List")
        for subtask in self.subtasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
            }.get(subtask.status, "❓")
            
            summary_lines.append(f"{status_icon} [{subtask.id}] {subtask.description}")
            
            if subtask.dependencies:
                deps_str = ", ".join(map(str, subtask.dependencies))
                summary_lines.append(f"    Dependencies: {deps_str}")
            
            if subtask.status == TaskStatus.COMPLETED and subtask.result:
                summary_lines.append(f"    Result: {subtask.result}")
            
            if subtask.status == TaskStatus.FAILED and subtask.error:
                summary_lines.append(f"    Error: {subtask.error}")
        
        summary_lines.append("")
        
        current = self.get_current_subtask()
        if current:
            summary_lines.append(f"## Current Subtask")
            summary_lines.append(f"[{current.id}] {current.description}")
        else:
            if self.is_all_completed():
                summary_lines.append("## All subtasks are completed!")
            else:
                summary_lines.append("## Waiting for subtasks to be executed...")
        
        return "\n".join(summary_lines)
    
    def reset(self):
        """Reset the planner state"""
        self.original_goal = None
        self.subtasks = []
        self.current_subtask_index = 0