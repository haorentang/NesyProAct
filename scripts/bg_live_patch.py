"""Runtime patch making browsergym 0.10.2 observation extraction robust on live websites.

Real sites inject dynamic iframes (ads, trackers) between the marking pass and the
recursive walk; such frames have no bid yet and the stock _pre_extract raises
MarkingError, killing the whole episode after 5 retries. Sandboxed WebArena sites
never trigger this. We skip unmarkable frames instead — they carry no task content.

Import this module BEFORE any browsergym env is created:
    import bg_live_patch  # noqa: F401
"""

import pkgutil
import re
import sys

# Stealth: substitute patchright (Playwright fork with automation leaks removed,
# the browser stack Mind2Web-2's authors specify) for playwright. The top-level
# alias makes every `import playwright.<submodule>` resolve inside patchright.
# Must run before browsergym (which imports playwright) is loaded.
if "playwright" not in sys.modules:
    try:
        import patchright
        sys.modules["playwright"] = patchright
    except ImportError:
        pass  # fall back to stock playwright

import browsergym.core.env as bg_env
import browsergym.core.observation as bg_obs


def _pre_extract_skip_unmarkable(page, tags_to_mark="standard_html"):
    js_frame_mark_elements = pkgutil.get_data(
        bg_obs.__name__, "javascript/frame_mark_elements.js").decode("utf-8")

    def mark_frames_recursive(frame, frame_bid):
        assert frame_bid == "" or re.match(r"^[a-z][a-zA-Z]*$", frame_bid)
        warning_msgs = frame.evaluate(
            js_frame_mark_elements, [frame_bid, bg_obs.BID_ATTR, tags_to_mark])
        for msg in warning_msgs:
            bg_obs.logger.warning(msg)

        for child_frame in frame.child_frames:
            if child_frame.is_detached():
                continue
            try:
                child_frame_elem = child_frame.frame_element()
            except Exception:
                continue  # frame vanished mid-walk
            if not child_frame_elem.content_frame() == child_frame:
                bg_obs.logger.warning(
                    f"Skipping frame '{child_frame.name}' for marking, seems problematic.")
                continue
            sandbox_attr = child_frame_elem.get_attribute("sandbox")
            if sandbox_attr is not None and "allow-scripts" not in sandbox_attr.split():
                continue
            child_frame_bid = child_frame_elem.get_attribute(bg_obs.BID_ATTR)
            if child_frame_bid is None:
                # dynamic frame injected after the marking pass (ads etc.) — skip it
                bg_obs.logger.warning("Skipping child frame without bid (live-web patch).")
                continue
            mark_frames_recursive(child_frame, frame_bid=child_frame_bid)

    mark_frames_recursive(page.main_frame, frame_bid="")


bg_obs._pre_extract = _pre_extract_skip_unmarkable
bg_env._pre_extract = _pre_extract_skip_unmarkable


# Patchright runs page.evaluate in an isolated world, so the chat window's
# rendering helper (addChatMessage) is unreachable. The message list itself is
# kept on the Python side (Chat.messages) and drives task validation; rendering
# is cosmetic, so render failures are safe to ignore.
import browsergym.core.chat as bg_chat  # noqa: E402

_orig_add_message = bg_chat.Chat.add_message


def _add_message_safe(self, role, msg):
    import time as _time
    try:
        return _orig_add_message(self, role, msg)
    except Exception:
        # upstream appends to self.messages before rendering, so the message is
        # normally stored already; append defensively if it is not
        stored = any(m.get("message") == msg and m.get("role") == role
                     for m in self.messages[-3:])
        if not stored and role in ("user", "user_image", "assistant", "infeasible"):
            self.messages.append(
                {"role": role, "timestamp": _time.time(), "message": msg})
        return None


bg_chat.Chat.add_message = _add_message_safe


# Headful browsers are required for stealth but must not pop over the user's
# screen: open every window far off-screen (they only appear in the taskbar).
# Patched at the BrowserType.launch level so it composes with browsergym's own
# args regardless of how the env is constructed.
import playwright.sync_api as _pw_sync  # resolves inside patchright when aliased

_orig_launch = _pw_sync.BrowserType.launch


# --- post-action settling -------------------------------------------------
# browsergym waits only for `domcontentloaded` + 0.5s before reading the next
# observation. Sandboxed WebArena pages are server-rendered, so that suffices;
# live sites update the DOM asynchronously after XHR, so the agent reads a
# stale tree, concludes its action had no effect, and retries the same click.
# We extend the existing wait hook (called before observation extraction) with
# a network-idle wait and a DOM-stability poll. Enabled by default; set
# OM2W_SETTLE=0 to restore stock behaviour.
import os as _os


def _wait_dom_loaded_settled(self):
    for page in self.context.pages:
        for state, timeout in (("domcontentloaded", 3000), ("networkidle", 3000)):
            try:
                page.wait_for_load_state(state, timeout=timeout)
            except Exception:
                pass  # trackers/polling keep some sites from ever going idle
        try:  # poll until the DOM stops growing (max ~1.5s)
            prev = -1
            for _ in range(5):
                cur = page.evaluate("document.body ? document.body.innerHTML.length : 0")
                if cur == prev:
                    break
                prev = cur
                page.wait_for_timeout(300)
        except Exception:
            pass
        for frame in page.frames:
            try:
                frame.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass


if _os.environ.get("OM2W_SETTLE", "1") != "0":
    bg_env.BrowserEnv._wait_dom_loaded = _wait_dom_loaded_settled


# --- recover the accessibility tree after client-side navigation ------------
# On SPA-style transitions the first extraction after the action returns a
# near-empty tree (the frame is re-created after browsergym marks the DOM);
# extracting once more returns the full page. Measured on craigslist search:
# 164 chars immediately after the action, 5934 chars on re-extraction.
# Without this the agent sees an empty page, concludes its click did nothing,
# and repeats it (up to 19 identical clicks observed in one task).
_MIN_AXTREE_CHARS = int(_os.environ.get("OM2W_MIN_AXTREE", 600))
_orig_get_obs = bg_env.BrowserEnv._get_obs


def _get_obs_retry_empty(self):
    obs = _orig_get_obs(self)
    try:
        from browsergym.utils.obs import flatten_axtree_to_str
        if len(flatten_axtree_to_str(obs.get("axtree_object", {}))) < _MIN_AXTREE_CHARS:
            for page in self.context.pages:
                try:
                    page.wait_for_timeout(500)
                except Exception:
                    pass
            obs = _orig_get_obs(self)
    except Exception:
        pass
    return obs


if _os.environ.get("OM2W_SETTLE", "1") != "0":
    bg_env.BrowserEnv._get_obs = _get_obs_retry_empty


_HIDE_TAG = "--om2w-automation"


def _hide_tagged_chrome_windows():
    """Hide (SW_HIDE) top-level windows of our tagged Chrome processes so they
    stay off the taskbar. Scoped by the launch tag, so the user's own Chrome is
    never touched. Anti-throttle flags keep hidden windows rendering, which
    screenshots depend on."""
    import ctypes
    import subprocess

    u32 = ctypes.windll.user32
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
             f"Where-Object {{ $_.CommandLine -match '{_HIDE_TAG[2:]}' }}).ProcessId"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.split()
    except Exception:
        return
    pids = {int(x) for x in out if x.isdigit()}
    if not pids:
        return

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _):
        pid = ctypes.c_ulong()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and u32.IsWindowVisible(hwnd):
            u32.ShowWindow(hwnd, 0)  # SW_HIDE
        return True

    u32.EnumWindows(_cb, 0)


def _launch_hidden(self, **kw):
    launch_args = list(kw.get("args") or [])
    for flag in (_HIDE_TAG,
                 "--disable-backgrounding-occluded-windows",
                 "--disable-renderer-backgrounding",
                 "--disable-features=CalculateNativeWinOcclusion"):
        if flag not in launch_args:
            launch_args.append(flag)
    kw["args"] = launch_args
    browser = _orig_launch(self, **kw)

    # windows appear a moment after launch, and new ones can open mid-episode
    import threading

    def _hider():
        for _ in range(40):  # ~2 min of coverage per browser
            _hide_tagged_chrome_windows()
            __import__("time").sleep(3)

    threading.Thread(target=_hider, daemon=True).start()
    return browser


_pw_sync.BrowserType.launch = _launch_hidden
