import re


def extract_all_elements(ax_tree: str) -> list[dict]:
    """
    Extract all the elements from the accessibility tree
    sample element object structure:
    {
        "id": "1",
        "role": "button",
        "value": "Click me",
        "attributes": [...]
    }
    Args:
        ax_tree: The accessibility tree of the web page
    Returns:
        A list of elements
    """
    elements = []
    lines = ax_tree.strip().split('\n')

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Remove leading tabs (indentation) - we don't need it for element extraction
        line = line.lstrip('\t')

        # Pattern to match: [ID] role 'value', attributes
        # Or: role 'value', attributes
        # Or: [ID] role '', attributes

        # First, try to extract ID if present
        element_id = None
        if line.startswith('['):
            id_match = re.match(r'\[(\d+)\]\s*(.+)', line)
            if id_match:
                element_id = id_match.group(1)
                line = id_match.group(2)

        # Now parse role, value, and attributes
        # Pattern: role 'value', attr1, attr2=value2
        # Or: role '', attr1, attr2=value2

        # Extract role (first word)
        role_match = re.match(r'^(\w+)(?:\s+(.+))?$', line)
        if not role_match:
            continue

        role = role_match.group(1)
        rest = role_match.group(2).strip() if role_match.group(2) else ""

        # Extract value (text in quotes)
        value = ""
        attributes = []

        # Check if there's a quoted value
        value_match = re.match(r"^'([^']*)'", rest)
        if value_match:
            value = value_match.group(1)
            rest = rest[len(value_match.group(0)):].strip()
        else:
            # Check for empty quotes
            if rest.startswith("''"):
                value = ""
                rest = rest[2:].strip()

        # Parse attributes (comma-separated)
        if rest:
            # Remove leading comma if present
            if rest.startswith(','):
                rest = rest[1:].strip()

            # Split by comma, but be careful with quoted values in attributes
            attr_parts = []
            current_part = ""
            in_quotes = False
            quote_char = None

            for char in rest:
                if char in ("'", '"') and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current_part += char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                    current_part += char
                elif char == ',' and not in_quotes:
                    if current_part.strip():
                        attr_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char

            if current_part.strip():
                attr_parts.append(current_part.strip())

            # Parse each attribute
            for attr_str in attr_parts:
                attr_str = attr_str.strip()
                if not attr_str:
                    continue

                # Check if it's a key=value attribute
                if '=' in attr_str:
                    key, val = attr_str.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    # Remove quotes from value if present
                    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    attributes.append(f"{key}='{val}'")
                else:
                    # It's a boolean attribute (like "focused")
                    attributes.append(attr_str)

        elements.append({
            "id": element_id,
            "role": role,
            "value": value,
            "attributes": attributes
        })

    return elements

def extract_element_by_id(ax_tree: str, element_id: str) -> str:
    """
    Extract the element from the accessibility tree by its ID
    Args:
        ax_tree: The accessibility tree of the web page
        element_id: The ID of the element to extract
    Returns:
        The line in the accessibiliy tree that represents the element
    """
    lines = ax_tree.strip().split('\n')

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Check if line starts with [element_id]
        # Pattern: [ID] followed by space and rest of content
        # Use regex to match the ID pattern, similar to extract_all_elements
        stripped_line = line.lstrip('\t')
        if stripped_line.startswith('['):
            id_match = re.match(r'\[(\d+)\]\s*(.+)', stripped_line)
            if id_match and id_match.group(1) == element_id:
                return line

    # Return empty string if element not found
    return ""

import re
from collections import deque
from typing import List, Dict, Any, Optional, Set


def extract_all_elements_with_context(
    ax_tree: str,
    *,
    max_ancestor: int = 4,
    max_sibling_text: int = 8,
    max_desc_text: int = 12,
    # Only descendants with these roles contribute option values (menu option semantics)
    desc_roles: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract all elements from AX tree and attach context:
      - depth/path
      - ancestors (up to max_ancestor)
      - left_sibling_text
      - descendant_text: ONLY collected from descendant nodes whose role in desc_roles
      - context_name: prefer descendant_text for empty-label parents

    element structure:
    {
        "id": "64",
        "role": "button",
        "value": "",
        "attributes": [...],
        "context": {...}
    }
    """
    if desc_roles is None:
        # option/tab/menuitem/listitem, etc.
        desc_roles = {"option", "tab", "menuitem", "listitem"}

    # normalize role set once
    desc_roles = {r.lower().strip() for r in desc_roles}

    elements: List[Dict[str, Any]] = []
    lines = ax_tree.strip().split("\n")

    stack: List[Dict[str, Any]] = []
    sibling_text_by_depth: Dict[int, deque] = {}

    def parse_line(raw_line: str):
        depth = 0
        while depth < len(raw_line) and raw_line[depth] == "\t":
            depth += 1

        line = raw_line.lstrip("\t").strip()
        if not line:
            return None

        element_id = None
        if line.startswith("["):
            id_match = re.match(r"\[(\d+)\]\s*(.+)", line)
            if id_match:
                element_id = id_match.group(1)
                line = id_match.group(2).strip()

        role_match = re.match(r"^(\w+)(?:\s+(.+))?$", line)
        if not role_match:
            return None
        role = role_match.group(1)
        rest = role_match.group(2).strip() if role_match.group(2) else ""

        value = ""
        attributes: List[str] = []

        value_match = re.match(r"^'([^']*)'", rest)
        if value_match:
            value = value_match.group(1)
            rest = rest[len(value_match.group(0)):].strip()
        else:
            if rest.startswith("''"):
                value = ""
                rest = rest[2:].strip()

        if rest:
            if rest.startswith(","):
                rest = rest[1:].strip()

            attr_parts = []
            current_part = ""
            in_quotes = False
            quote_char = None

            for char in rest:
                if char in ("'", '"') and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current_part += char
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                    current_part += char
                elif char == "," and not in_quotes:
                    if current_part.strip():
                        attr_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char

            if current_part.strip():
                attr_parts.append(current_part.strip())

            for attr_str in attr_parts:
                attr_str = attr_str.strip()
                if not attr_str:
                    continue
                if "=" in attr_str:
                    key, val = attr_str.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    attributes.append(f"{key}='{val}'")
                else:
                    attributes.append(attr_str)

        return element_id, role, value, attributes, depth

    def make_context_name(
        role: str,
        value: str,
        sibling_text: List[str],
        ancestors: List[Dict[str, str]],
        descendant_text: List[str],
    ) -> str:
        if value.strip():
            return value

        # Prefer descendant option text (best for menu buttons and comboboxes)
        desc_hint = " | ".join([t for t in descendant_text if t.strip()][:3])
        if desc_hint:
            return f"{role} with options: {desc_hint}"

        sib_hint = " | ".join([t for t in sibling_text if t.strip()][:3])
        if sib_hint:
            return f"{role} near: {sib_hint}"

        anc_hint = " > ".join([a["value"] for a in ancestors if a.get("value")][:2])
        if anc_hint:
            return f"{role} in: {anc_hint}"

        return f"{role}"

    def flush_popped_nodes(target_depth: int):
        while len(stack) > target_depth:
            node = stack.pop()
            desc = node.get("_desc_buf", [])
            if desc:
                seen = set()
                uniq = []
                for t in desc:
                    if t not in seen:
                        uniq.append(t)
                        seen.add(t)
                    if len(uniq) >= max_desc_text:
                        break
                node["context"]["descendant_text"] = uniq
                node["context"]["descendant_text_short"] = " | ".join(uniq[:5])
            else:
                node["context"]["descendant_text"] = []
                node["context"]["descendant_text_short"] = ""

    for raw in lines:
        if not raw.strip():
            continue

        parsed = parse_line(raw)
        if not parsed:
            continue
        element_id, role, value, attributes, depth = parsed
        role_norm = (role or "").lower().strip()

        flush_popped_nodes(depth)

        parent = stack[-1] if stack else None

        ancestors = []
        for a in stack[-max_ancestor:]:
            ancestors.append({
                "id": a.get("id"),
                "role": a.get("role"),
                "value": a.get("value"),
            })

        dq = sibling_text_by_depth.setdefault(depth, deque(maxlen=max_sibling_text))
        left_sibling_text = list(dq)

        elem: Dict[str, Any] = {
            "id": element_id,
            "role": role,
            "value": value,
            "attributes": attributes,
            "context": {
                "depth": depth,
                "parent_id": parent.get("id") if parent else None,
                "ancestors": ancestors,
                "left_sibling_text": left_sibling_text,
                "path": ".".join([(a.get("id") or a.get("role") or "x") for a in ancestors] +
                                 [(element_id or role or "x")]),
            },
            "_desc_buf": []
        }

        # Only descendants with the listed roles contribute option semantics
        if value and value.strip() and (role_norm in desc_roles):
            token = value.strip()
            for anc in stack:
                buf = anc.setdefault("_desc_buf", [])
                # collect a few extra; deduplicated and truncated later
                if len(buf) < max_desc_text * 2:
                    buf.append(token)

        # sibling buffer: only non-empty values are recorded, used as nearby hints
        if value and value.strip():
            dq.append(value.strip())

        stack.append(elem)
        elements.append(elem)

    flush_popped_nodes(0)

    # finalize context_name after descendant_text is ready
    for e in elements:
        desc = (e.get("context") or {}).get("descendant_text", [])
        e["context"]["context_name"] = make_context_name(
            role=e.get("role") or "",
            value=e.get("value") or "",
            sibling_text=(e.get("context") or {}).get("left_sibling_text", []),
            ancestors=(e.get("context") or {}).get("ancestors", []),
            descendant_text=desc,
        )
        e.pop("_desc_buf", None)

    return elements