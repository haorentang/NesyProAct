import re
from typing import Any, Dict, List, Tuple, Set

from interface.utils import extract_all_elements_with_context

_attr_kv_re = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*=\s*(.+?)\s*$")

def parse_attributes(attr_list: List[str]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for raw in (attr_list or []):
        s = raw.strip()
        if not s:
            continue
        m = _attr_kv_re.match(s)
        if not m:
            attrs[s] = True
            continue
        k, v = m.group(1), m.group(2).strip()
        if (len(v) >= 2) and (v[0] == v[-1]) and v[0] in {"'", '"'}:
            attrs[k] = v[1:-1]
        elif v in {"True", "False"}:
            attrs[k] = (v == "True")
        else:
            attrs[k] = v
    return attrs

def norm(s: str) -> str:
    return (s or "").lower().strip()


def elem_text(e: Dict[str, Any]) -> str:
    """
    LLM-friendly retrieval text:
    value + context_name + sibling text + (menu options / descendant text)
    """
    v = norm(e.get("value", ""))
    ctx = e.get("context") or {}
    cn = norm(ctx.get("context_name", ""))

    sib = " ".join([norm(x) for x in (ctx.get("left_sibling_text") or [])[-5:]])

    # Option texts from the subtree (values of descendant menuitem/option/tab/listitem)
    desc_list = ctx.get("descendant_text") or []
    # Keep only the first few to bound the length
    desc = " ".join([norm(x) for x in desc_list[:6] if str(x).strip()])

    return " ".join([t for t in [v, cn, sib, desc] if t]).strip()


def passes_basic_state(attrs: Dict[str, Any]) -> bool:
    if attrs.get("disabled") is True:
        return False
    if attrs.get("enabled") is False:
        return False
    if attrs.get("visible") is False:
        return False
    return True


ACTION_TO_ROLES = {
    # Plain interaction: click a concrete clickable target
    "click": {"link", "button", "menuitem", "listitem", "option", "tab"},

    # Text input
    "fill":  {"searchbox", "textbox", "combobox"},

    # Hover (mostly to open a menu/popup)
    "hover": {"link", "button", "menuitem", "listitem", "tab"},

    # Key press to confirm (e.g., Enter to search)
    "press": {"searchbox", "textbox", "combobox"},

    # Choose from a control that holds a set of options
    "select_option": {
        "button",      # button with hasPopup=menu
        "combobox",    # dropdown
        "listbox",     # listbox with option children
        "tablist",     # tab container (with tab options)
        "menu",        # explicit menu container
    },
}


def filter_by_action(elements: List[Dict[str, Any]], action_type: str) -> List[Dict[str, Any]]:
    roles = ACTION_TO_ROLES.get(action_type, set())
    out = []

    for e in elements:
        role = norm(e.get("role", ""))

        # 1) role filter (generic)
        if roles and role not in roles:
            continue

        attrs = parse_attributes(e.get("attributes") or [])
        if not passes_basic_state(attrs):
            continue

        # 2) action-specific gates
        if action_type == "select_option":
            ctx = e.get("context") or {}
            desc = ctx.get("descendant_text") or []

            # 2.1 Must show a signal of holding a set of options
            # - combobox/listbox/tablist/menu are option containers by nature
            # - a button counts only if it is a menu button (hasPopup=menu)
            is_container_role = role in {"combobox", "listbox", "tablist", "menu"}
            is_menu_button = (role == "button" and attrs.get("hasPopup") == "menu")

            if not (is_container_role or is_menu_button):
                continue

            # 2.2 Must actually have options (values of the extracted descendant roles)
            if not desc:
                continue

        e2 = dict(e)
        e2["attrs"] = attrs
        out.append(e2)

    return out


INTENT_KEYWORDS = {
    "Search":  ["search", "find", "query", "lookup"],
    "Filter":  ["filter", "refine", "narrow"],
    "Sort":    ["sort", "order", "newest", "latest", "oldest"],
    "Submit":  ["submit", "post", "publish", "create", "save", "confirm", "continue"],
    "Add":     ["add", "create", "new", "follow", "subscribe", "join"],
    "Remove":  ["remove", "delete", "clear", "unfollow", "unsubscribe", "leave"],
    "Navigate":["home", "back", "next", "previous", "menu"],
    "Select":  ["select", "choose", "open", "view"],
}

STOPWORDS = {
    "navigate","go","open","to","the","a","an","page","section","tab","view",
    "please","click","and","then","on","in","of","for","my","your"
}

def tokenize(s: str):
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    toks = [t for t in toks if len(t) >= 2]   # drop single-character tokens
    return [t for t in toks if t not in STOPWORDS]


def token_set(s: str) -> Set[str]:
    return set(tokenize(s))

def query_match_score(query: str, text: str) -> float:
    """
    Return a similarity score in [0, 1+], used as the main ranking signal.
    Coverage (how many query tokens are matched) is weighted more than precision.
    """
    q = token_set(query)
    t = token_set(text)
    if not q or not t:
        return 0.0

    inter = q & t
    if not inter:
        return 0.0

    # coverage: fraction of query tokens matched (main signal)
    coverage = len(inter) / len(q)

    # precision: fraction of text tokens that come from the query (penalizes generic text)
    precision = len(inter) / len(t)

    # F-like combination, weighting coverage more
    return 1.6 * coverage + 0.6 * precision


def score_by_intent(e: Dict[str, Any], intent_label: str, query: str = "") -> float:
    role = norm(e.get("role", ""))
    attrs = e.get("attrs") or {}
    text = elem_text(e)

    kws = [norm(k) for k in INTENT_KEYWORDS.get(intent_label, [])]
    s = 0.0

    # 1) query match (strongest signal)
    if query:
        s += 3.0 * query_match_score(query, text)

    # 2) intent keyword match (weak signal)
    if kws and text:
        if any(k in text for k in kws):
            s += 0.8

    # 3) intent x role preference (very light)
    if intent_label == "Search":
        s += {"searchbox": 1.2, "textbox": 0.8, "combobox": 0.7}.get(role, 0.0)
    elif intent_label == "Filter":
        s += {"button": 0.6, "searchbox": 0.9, "textbox": 0.6, "combobox": 0.6}.get(role, 0.0)
    elif intent_label == "Sort":
        s += {"button": 0.9, "combobox": 0.8, "option": 0.7, "menuitem": 0.7, "link": 0.4}.get(role, 0.0)
    elif intent_label == "Navigate":
        s += {"link": 0.7, "tab": 0.6, "menuitem": 0.6, "button": 0.3}.get(role, 0.0)
    elif intent_label in {"Submit", "Add", "Remove"}:
        s += {"button": 0.8, "link": 0.5, "menuitem": 0.6}.get(role, 0.0)
    elif intent_label == "Select":
        s += {"link": 0.7, "button": 0.5, "menuitem": 0.6, "option": 0.6, "listitem": 0.2}.get(role, 0.0)

    # 4) popup/menu signal (common for Filter/Sort)
    if intent_label in {"Filter", "Sort"} and attrs.get("hasPopup") == "menu":
        s += 1.5
        if attrs.get("expanded") is False:
            s += 0.2

    # 4.5) bonus for containers holding a set of options (useful for Filter/Sort/Select)
    desc = (e.get("context") or {}).get("descendant_text") or []
    if desc and intent_label in {"Filter", "Sort", "Select"}:
        s += 0.4
        if attrs.get("hasPopup") == "menu":
            s += 0.6

    # 5) focused/selected (weak)
    if attrs.get("focused") is True:
        s += 0.2
    if attrs.get("selected") is True:
        s += 0.2

    # 6) de-noise listitems using query token overlap rather than full-string containment
    if role == "listitem":
        if (not query) or (query_match_score(query, text) < 0.15):
            s -= 0.3

    return s


def retrieve_topk(elements: List[Dict[str, Any]],
                  action_type: str,
                  intent_label: str,
                  query: str = "",
                  topk: int = 20) -> List[Dict[str, Any]]:
    # 1) action filter
    cands = filter_by_action(elements, action_type)

    # 2) intent rank
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in cands:
        s = score_by_intent(e, intent_label, query=query)
        scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 3) output (keep context and parsed attributes)
    out = []
    for s, e in scored[:topk]:
        out.append({
            "score": round(s, 3),
            "id": e.get("id"),
            "role": e.get("role"),
            "value": e.get("value"),
            "attributes": e.get("attributes"),
            "attrs": e.get("attrs"),
            "context": e.get("context"),
        })
    return out

def retrieve_click(elements, intent_label, query="", topk=10):
    return retrieve_topk(elements, "click", intent_label, query=query, topk=topk)

def retrieve_fill(elements, intent_label, query="", topk=10):
    return retrieve_topk(elements, "fill", intent_label, query=query, topk=topk)

def retrieve_hover(elements, intent_label, query="", topk=10):
    return retrieve_topk(elements, "hover", intent_label, query=query, topk=topk)

def retrieve_press(elements, intent_label, query="", topk=10):
    return retrieve_topk(elements, "press", intent_label, query=query, topk=topk)

def retrieve_select_option(
    elements,
    intent_label,
    query="",
    option_query="",   # optional: a specific option text extracted from the plan
    topk=10
):
    """
    Retrieve container elements suitable for select_option(bid, options).

    Steps:
    1) filter by action type = select_option
    2) rank by intent + query similarity
    3) optionally refine by matching option_query against descendant_text
    """

    # 1) action filter (only option containers)
    cands = filter_by_action(elements, "select_option")

    if not cands:
        return []

    # 2) intent + query rank
    scored = []
    for e in cands:
        s = score_by_intent(e, intent_label, query=query)
        scored.append((s, e))

    scored.sort(key=lambda x: x[0], reverse=True)

    ranked = [e for s, e in scored]

    # 3) optional refinement: must contain desired option
    if option_query:
        oq = norm(option_query)
        refined = []
        for e in ranked:
            desc = (e.get("context") or {}).get("descendant_text") or []
            hay = " ".join(norm(x) for x in desc)
            if oq in hay:
                refined.append(e)

        # fall back to the original ranking if refinement removes every candidate
        if refined:
            ranked = refined

    # 4) output format
    out = []
    for e in ranked[:topk]:
        out.append({
            "id": e.get("id"),
            "role": e.get("role"),
            "value": e.get("value"),
            "attributes": e.get("attributes"),
            "attrs": e.get("attrs"),
            "context": e.get("context"),
        })

    return out


KEY_ATTRS = ["hasPopup", "expanded", "focused", "selected", "checked", "pressed"]

def _pick_label(e: Dict[str, Any]) -> str:
    v = (e.get("value") or "").strip()
    if v:
        return v
    ctx = e.get("context") or {}
    cn = (ctx.get("context_name") or "").strip()
    return cn or ""

def _pick_attrs(e: Dict[str, Any]) -> str:
    attrs = e.get("attrs") or {}
    keep = {k: attrs[k] for k in KEY_ATTRS if k in attrs}
    if not keep:
        return "{}"
    # compact formatting
    inner = ", ".join([f"{k}:{str(v).lower() if isinstance(v,bool) else v}" for k,v in keep.items()])
    return "{" + inner + "}"


def _pick_ctx(e: Dict[str, Any]) -> str:
    ctx = e.get("context") or {}

    ancestors = ctx.get("ancestors") or []
    anc_short = []
    for a in ancestors[-2:]:
        r = a.get("role") or ""
        val = (a.get("value") or "").strip()
        anc_short.append(r if not val else f"{r}:{val}")

    sib = ctx.get("left_sibling_text") or []
    sib_short = [str(s).strip() for s in sib[-4:] if str(s).strip()]

    # For select_option, the option texts are the key semantics
    desc = ctx.get("descendant_text") or []
    opt_short = [str(x).strip() for x in desc[:8] if str(x).strip()]  # keep the first 8 to bound the length

    parts = []
    if anc_short:
        parts.append("ancestors=[" + " > ".join(anc_short) + "]")
    if sib_short:
        parts.append('siblings=["' + '","'.join(sib_short) + '"]')
    if opt_short:
        parts.append('options=["' + '","'.join(opt_short) + '"]')

    return "  ".join(parts) if parts else ""


def format_candidates_for_llm(cands: List[Dict[str, Any]], action: str, intent: str) -> str:
    lines = [f"[Candidate Elements] (action={action}, intent={intent}, topk={len(cands)})"]
    for e in cands:
        eid = e.get("id")
        role = e.get("role")
        label = _pick_label(e)
        attrs = _pick_attrs(e)
        ctx = _pick_ctx(e)

        lines.append(f"#{eid}  role={role}  label=\"{label}\"  attrs={attrs}")

        if ctx:
            lines.append(f"     ctx: {ctx}")

        # For select_option, also print the options on a separate line
        if action == "select_option":
            desc = (e.get("context") or {}).get("descendant_text") or []
            opt_short = [str(x).strip() for x in desc[:10] if str(x).strip()]
            if opt_short:
                lines.append(f"     available_options: {opt_short}")

    return "\n".join(lines)
