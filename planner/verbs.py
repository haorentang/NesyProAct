"""Subgoal verbs available to the decomposition layer."""
from typing import List, Dict, Any


SEARCH = {
    "name": "search",
    "role": "Search for items using a search input box by entering keywords",
    "allowed_intents": ["Search"],
    "example": "search for machine learning books",
    "notes": [
        "Use this when a search input box is the way to find the items",
    ],
}

FIND = {
    "name": "find",
    "role": "Find ONE specific item on the current page",
    "example": "find the opening hours of the library",
    "notes": ["Do not use this verb when you need to list several items"],
}

NAVIGATE = {
    "name": "navigate",
    "role": "Navigate to a specific page or section by clicking links, buttons, or menu items",
    "forbidden_intents": ["Search"],
    "example": "navigate to the account page",
    "notes": [
        "Use this when you need to click on links, buttons, or menu items to go to another page or section",
        "This is different from 'search', which uses a search input box",
    ],
}

LIST = {
    "name": "list",
    "role": "List the specified items (following pagination if needed) and send the findings to the user",
    "example": "list the titles of all books on the first results page",
    "notes": [
        "Use this to report requested information",
        "Specify what you want to list",
    ],
}

SORT = {
    "name": "sort",
    "role": "Sort items by a specific criterion (e.g., price, rating, date)",
    "example": "sort results by publication date",
    "forbidden_intents": ["Extract", "Navigate"],
    "notes": [],
}

SUBMIT = {
    "name": "submit",
    "role": "Submit a form or complete an action",
    "example": "submit the registration form",
    "notes": [],
}

ADD = {
    "name": "add",
    "role": "Add an item to a list or collection",
    "example": "add the book to the reading list",
    "notes": [],
}

REMOVE = {
    "name": "remove",
    "role": "Remove an item from a list or collection",
    "example": "remove the book from the reading list",
    "notes": [],
}

EDIT = {
    "name": "edit",
    "role": "Edit or modify content or settings",
    "example": "edit the profile description",
    "notes": [],
}

DELETE = {
    "name": "delete",
    "role": "Delete an item or content",
    "example": "delete a saved search",
    "notes": [],
}

COMPARE = {
    "name": "compare",
    "role": "Compare multiple items or options",
    "example": "compare the two editions",
    "notes": [],
}


VERB_DEFINITIONS = {
    "search": SEARCH,
    "navigate": NAVIGATE,
    "find": FIND,
    "list": LIST,
    "sort": SORT,
    "submit": SUBMIT,
    "add": ADD,
    "remove": REMOVE,
    "edit": EDIT,
    "delete": DELETE,
    "compare": COMPARE,
}


def get_verb_descriptions_str() -> str:
    """Format all verb descriptions for the planner prompt."""
    lines = []
    for verb_def in VERB_DEFINITIONS.values():
        notes = f", notes: {verb_def['notes']}" if verb_def["notes"] else ""
        lines.append(f"- {verb_def['name']}: {verb_def['role']} (e.g., {verb_def['example']}{notes})")
    return "\n".join(lines)


VERB_DESCRIPTIONS_STR = get_verb_descriptions_str()


def extract_verbs_from_description(description: str) -> List[str]:
    """Return the verbs from VERB_DEFINITIONS that appear as whole words in a subtask description."""
    description_lower = description.lower()
    found_verbs = []
    for verb in VERB_DEFINITIONS:
        if (description_lower.startswith(verb + " ")
                or f" {verb} " in description_lower
                or description_lower == verb):
            found_verbs.append(verb)
    return found_verbs


def get_intents_and_roles_for_verbs(verbs: List[str]) -> Dict[str, Any]:
    """Collect allowed intents, forbidden intents, and roles for a list of verbs."""
    allowed_intents, forbidden_intents, roles = [], [], []
    for verb in verbs:
        verb_def = VERB_DEFINITIONS.get(verb)
        if not verb_def:
            continue
        allowed_intents.extend(verb_def.get("allowed_intents", []))
        forbidden_intents.extend(verb_def.get("forbidden_intents", []))
        if "role" in verb_def:
            roles.append(verb_def["role"])
    return {
        "allowed_intents": list(dict.fromkeys(allowed_intents)),
        "forbidden_intents": list(dict.fromkeys(forbidden_intents)),
        "roles": roles,
    }
