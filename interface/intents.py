INTENT_LABEL = {
    "Navigate": {
        "description": "Navigate to a specific page or section of the website.",
        "allowed_actions": ["click"]
    },
    "Search": {
        "description": "Search for a specific item or information on the website.",
        "allowed_actions": ["fill", "click", "press"]
    },
    "Filter": {
        "description": "Filter the listed items on the website.",
        "allowed_actions": ["fill", "click", "press", "select_option"]
    },
    "Sort": {
        "description": "Sort the listed items on the website by a specific criterion.",
        "allowed_actions": ["click", "select_option"]
    },
    "Select": {
        "description": "Select a specific item or information on the website with no page transition.",
        "allowed_actions": ["click", "select_option"]
    },
    "Submit": {
        "description": "Submit a form on the website.",
        "allowed_actions": ["click"]
    },
    "Add": {
        "description": "Add a new item on the website to some list, e.g. to a cart.",
        "allowed_actions": ["click"]
    },
    "Remove": {
        "description": "Remove an item from a list on the website, e.g. from a cart.",
        "allowed_actions": ["click"]
    },
    "Extract": {
        "description": "Extract required information from the website.",
        "allowed_actions": ["store", "send_msg_to_user"]
    },
    "Change_page": {
        "description": "Change to another page by clicking on the page number",
        "allowed_actions": ["click"]
    },
    "Write": {
        "description": "Write the content to the website, e.g. to a text area or a comment field.",
        "allowed_actions": ["fill", "press"]
    }
}

