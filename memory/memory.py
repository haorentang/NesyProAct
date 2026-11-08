"""
Memory System - Stores execution experiences

Each memory entry: m = (q, steps, k, R)
- q: subtask query
- steps: list of execution steps, each step contains:
    - action_type: type of action (e.g., "click", "fill", "goto")
    - action_str: full action string (e.g., "click('123')")
- k: insight operator (None for cold execution)
- R: binary outcome (0=failure, 1=success) - judged by LLM at subtask level
"""
import dataclasses
from typing import List, Optional, Dict, Any
from enum import Enum


class ExecutionOutcome(Enum):
    """Execution outcome"""
    SUCCESS = 1
    FAILURE = 0


@dataclasses.dataclass
class ExecutionStep:
    """A single execution step"""
    step_index: int  # Step index (0-based)
    action_type: str  # Action type (e.g., "click", "fill", "goto")
    action_str: str  # Full action string (e.g., "click('123')")
    semantic_target: Optional[str] = None  # Semantic name (e.g., "search_input")
    target_features: Optional[Dict[str, Any]] = None  # Element features (role, name, aria_label, etc.)
    original_bid: Optional[str] = None  # Original bid (for fallback)
    action_reason: Optional[str] = None  # Reason for choosing this action (from LLM)
    action_context: Optional[Dict[str, Any]] = None  # Context information around the action node
    error_msg: Optional[str] = None  # Error message if action execution failed
    url: Optional[str] = None  # Page URL at this step
    axtree: Optional[str] = None  # AXTree string after this step execution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "step_index": self.step_index,
            "action_type": self.action_type,
            "action_str": self.action_str
        }
        if self.semantic_target:
            result["semantic_target"] = self.semantic_target
        if self.target_features:
            result["target_features"] = self.target_features
        if self.original_bid:
            result["original_bid"] = self.original_bid
        if self.action_reason:
            result["action_reason"] = self.action_reason
        if self.action_context:
            result["action_context"] = self.action_context
        if self.error_msg:
            result["error_msg"] = self.error_msg
        if self.url:
            result["url"] = self.url
        if self.axtree:
            result["axtree"] = self.axtree
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        """Create from dictionary"""
        return cls(
            step_index=data["step_index"],
            action_type=data["action_type"],
            action_str=data["action_str"],
            semantic_target=data.get("semantic_target"),
            target_features=data.get("target_features"),
            original_bid=data.get("original_bid"),
            action_reason=data.get("action_reason"),
            action_context=data.get("action_context"),
            error_msg=data.get("error_msg"),
            url=data.get("url"),
            axtree=data.get("axtree")
        )


@dataclasses.dataclass
class MemoryEntry:
    """Memory entry storing execution experience"""
    query: str  # Subtask query q
    steps: List[ExecutionStep]  # List of execution steps
    operator: Optional[str] = None  # Insight operator k (None for cold execution)
    outcome: ExecutionOutcome = ExecutionOutcome.FAILURE  # Execution outcome R
    yaml_action_sequence: Optional[List[Dict[str, Any]]] = None  # SkillManager YAML action sequence (for reuse)
    skill_name: Optional[str] = None  # Generalized skill name (e.g., "search_item" for "search for machine learning books")
    skill_description: Optional[str] = None  # One-sentence description of the skill
    skill_parameter: Optional[List[str]] = None  # List of parameter names for the skill (e.g., ["item_name"])
    url_before: Optional[str] = None  # URL before subtask execution
    
    # Legacy support: skill field (deprecated, use steps instead)
    @property
    def skill(self) -> List[str]:
        """Legacy property: extract skill from steps"""
        return [step.action_str for step in self.steps]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "operator": self.operator,
            "outcome": self.outcome.value,
            # Legacy support
            "skill": self.skill
        }
        if self.yaml_action_sequence is not None:
            result["yaml_action_sequence"] = self.yaml_action_sequence
        if self.skill_name is not None:
            result["skill_name"] = self.skill_name
        if self.skill_description is not None:
            result["skill_description"] = self.skill_description
        if self.skill_parameter is not None:
            result["skill_parameter"] = self.skill_parameter
        if self.url_before is not None:
            result["url_before"] = self.url_before
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary"""
        # Support both new format (with steps) and legacy format (with skill)
        if "steps" in data:
            steps = [ExecutionStep.from_dict(step_data) for step_data in data["steps"]]
        else:
            # Legacy format: convert skill to steps
            steps = []
            skill = data.get("skill", [])
            for i, action_str in enumerate(skill):
                # Extract action type from action string
                action_type = cls._extract_action_type(action_str)
                steps.append(ExecutionStep(
                    step_index=i,
                    action_type=action_type,
                    action_str=action_str
                ))
        
        return cls(
            query=data.get("query", ""),  # Support legacy format without query
            steps=steps,
            operator=data.get("operator"),
            outcome=ExecutionOutcome(data["outcome"]),
            yaml_action_sequence=data.get("yaml_action_sequence"),
            skill_name=data.get("skill_name"),
            skill_description=data.get("skill_description"),
            skill_parameter=data.get("skill_parameter"),
            url_before=data.get("url_before")
        )
    
    @staticmethod
    def _extract_action_type(action_str: str) -> str:
        """Extract action type from action string"""
        import re
        match = re.match(r"(\w+)\(", action_str)
        if match:
            return match.group(1)
        return "unknown"


class Memory:
    """
    Memory system that stores and retrieves execution experiences
    """
    
    def __init__(self):
        """Initialize empty memory"""
        self.entries: List[MemoryEntry] = []
    
    def add(self, entry: MemoryEntry):
        """Add a memory entry"""
        self.entries.append(entry)
    
    def find_successful_skill(
        self, 
        query: str
    ) -> Optional[MemoryEntry]:
        """
        Find a successful skill for the given query
        
        Returns:
            MemoryEntry if found, None otherwise
        """
        for entry in self.entries:
            if (entry.query == query and 
                entry.outcome == ExecutionOutcome.SUCCESS):
                return entry
        return None
    
    def find_failed_attempts(
        self,
        query: str
    ) -> List[MemoryEntry]:
        """
        Find all failed attempts for the given query
        
        Returns:
            List of failed memory entries
        """
        return [
            entry for entry in self.entries
            if (entry.query == query and 
                entry.outcome == ExecutionOutcome.FAILURE)
        ]
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries"""
        return self.entries.copy()
    
    def clear(self):
        """Clear all memory entries"""
        self.entries.clear()
    
    def size(self) -> int:
        """Get the number of memory entries"""
        return len(self.entries)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        return {
            "entries": [entry.to_dict() for entry in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary"""
        memory = cls()
        for entry_data in data.get("entries", []):
            memory.add(MemoryEntry.from_dict(entry_data))
        return memory
