"""Log in to the WebArena sites inside a BrowserGym environment.

Site URLs are read from the standard WebArena environment variables (WA_SHOPPING, WA_SHOPPING_ADMIN,
WA_REDDIT, WA_GITLAB). The accounts below are the default test accounts shipped with the WebArena
Docker images (https://github.com/web-arena-x/webarena).
"""
import logging
import os
import time
import traceback

from browsergym.utils.obs import flatten_axtree_to_str
from interface.utils import extract_all_elements

logger = logging.getLogger(__name__)

SITE_URLS = {
    "shopping": os.environ.get("WA_SHOPPING", ""),
    "shopping_admin": os.environ.get("WA_SHOPPING_ADMIN", ""),
    "reddit": os.environ.get("WA_REDDIT", ""),
    "gitlab": os.environ.get("WA_GITLAB", ""),
}

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

LOGIN_PAGES = {
    "shopping": {
        "path": "/customer/account/login/",
        "username_search": ["Email", "email"],
        "password_search": ["Password", "password"],
        "button_search": ["Sign In", "Sign in", "Login"],
    },
    "reddit": {
        "path": "/login",
        "username_search": ["Username", "username"],
        "password_search": ["Password", "password"],
        "button_search": ["Log in", "Log In", "Login"],
    },
    "shopping_admin": {
        "path": "",
        "username_search": ["user name", "username", "User Name"],
        "password_search": ["password", "Password"],
        "button_search": ["Sign in", "Sign In", "Login"],
    },
    "gitlab": {
        "path": "/users/sign_in",
        "username_search": ["username", "Username"],
        "password_search": ["password", "Password"],
        "button_search": ["Sign in", "Sign In", "sign-in"],
    },
}


def locate_element_by_value(elements: list[dict], search_values: list[str], role: str = None) -> str:
    """Return the bid of the first element (optionally of a given role) whose value contains a search value."""
    for search_value in search_values:
        for elem in elements:
            if role and elem.get("role") != role:
                continue
            if search_value.lower() in elem.get("value", "").lower() and elem.get("id"):
                return str(elem["id"])
    return None


def login_site_with_env(env, site: str) -> bool:
    """Log in to one WebArena site using the BrowserGym environment. Returns True on success."""
    if site not in LOGIN_PAGES or not SITE_URLS.get(site):
        logger.warning(f"No login configuration (or WA_* URL) for site: {site}")
        return False

    page = LOGIN_PAGES[site]
    account = ACCOUNTS[site]
    login_url = f"{SITE_URLS[site]}{page['path']}"
    try:
        logger.info(f"Navigating to login page: {login_url}")
        obs, _, _, _, _ = env.step(f"goto('{login_url}')")
        time.sleep(0.2)

        elements = extract_all_elements(flatten_axtree_to_str(obs.get("axtree_object", {})))

        # Username field: a textbox whose label matches; otherwise the first textbox on the page
        username_bid = locate_element_by_value(elements, page["username_search"], role="textbox")
        if not username_bid:
            username_bid = next((e.get("id") for e in elements if e.get("role") == "textbox"), None)
        if not username_bid:
            logger.error(f"Failed to locate username field for {site}")
            return False

        # Password field: an input with type=password; otherwise match by label
        password_bid = None
        for elem in elements:
            if elem.get("role") != "textbox":
                continue
            for attr in elem.get("attributes", []):
                if isinstance(attr, dict) and attr.get("name") == "type" and attr.get("value") == "password":
                    password_bid = elem.get("id")
                    break
            if password_bid:
                break
        if not password_bid:
            password_bid = locate_element_by_value(elements, page["password_search"], role="textbox")
        if not password_bid:
            logger.error(f"Failed to locate password field for {site}")
            return False

        button_bid = locate_element_by_value(elements, page["button_search"], role="button")
        if not button_bid:
            logger.error(f"Failed to locate login button for {site}")
            return False

        env.step(f"fill('{username_bid}', '{account['username']}')")
        env.step(f"fill('{password_bid}', '{account['password']}')")
        obs, _, _, _, _ = env.step(f"click('{button_bid}')")
        time.sleep(0.5)
        logger.info(f"Login completed, current URL: {obs.get('url', '')}")
        return True
    except Exception as e:
        logger.error(f"Failed to log in to {site}: {e}")
        logger.error(traceback.format_exc())
        return False
