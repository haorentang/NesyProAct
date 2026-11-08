"""Generate per-task WebArena configs.

Reads WebArena's `test.raw.json` (from https://github.com/web-arena-x/webarena/tree/main/config_files),
replaces the site placeholders with the URLs of your WebArena deployment, and writes one JSON file per
task to `configs/tasks/{task_id}.json`.

Required environment variables (same names as BrowserGym's WebArena integration):
    WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_MAP, WA_WIKIPEDIA

Usage:
    python configs/generate_task_configs.py --raw path/to/test.raw.json
"""
import argparse
import json
import os
from pathlib import Path

PLACEHOLDERS = {
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",  # must come before __SHOPPING__
    "__SHOPPING__": "WA_SHOPPING",
    "__REDDIT__": "WA_REDDIT",
    "__GITLAB__": "WA_GITLAB",
    "__MAP__": "WA_MAP",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="Path to WebArena's test.raw.json")
    parser.add_argument("--out_dir", default=str(Path(__file__).parent / "tasks"))
    args = parser.parse_args()

    raw = Path(args.raw).read_text(encoding="utf-8")
    for placeholder, env_var in PLACEHOLDERS.items():
        if placeholder in raw:
            url = os.environ.get(env_var)
            if not url:
                raise SystemExit(f"Environment variable {env_var} is not set (needed for {placeholder}).")
            raw = raw.replace(placeholder, url.rstrip("/"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(raw)
    for task in tasks:
        with open(out_dir / f"{task['task_id']}.json", "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)
    print(f"Wrote {len(tasks)} task configs to {out_dir}")


if __name__ == "__main__":
    main()
