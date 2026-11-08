# NesyProAct: Proactive Neural-Symbolic Control for Web Agents

Official code for **"NesyProAct: Proactive Neural-Symbolic Control for Web Agents"** (NeurIPS 2026).

NesyProAct turns a ReAct-style web agent into an execution-aware decision process. Instead of assuming
that every click or keystroke had its intended effect, the agent states, before each step, what should
become observable on the page if the step succeeds, and checks it on the live page after acting.
Failed checks trigger targeted repair, and successful subgoal executions are distilled into reusable
skills whose steps are re-grounded on the current page when replayed.

## How it works

| Layer | What it does | Code |
|---|---|---|
| Decomposition | Splits the task into subgoal intents with precondition predicates | `planner/` |
| Transition | Synthesizes an action-query program for the current page | `orchestrator/skill_manager.py` |
| State | Ranks candidate UI elements for each query on the accessibility tree | `interface/state_registry.py` |
| Action | Grounds each action to an element and executes it | `orchestrator/skill_manager.py` |
| Verification | Judges whether a subgoal was reached; drives repair | `orchestrator/skill_manager.py`, `orchestrator/skill_executor.py` |
| Skills | Induces reusable skills and replays them with per-step re-grounding | `orchestrator/llm_judge.py`, `orchestrator/skill_executor.py`, `orchestrator/utils.py` |

All prompts are in `planner/prompt.py` and `orchestrator/prompts.py`.

## Installation

```bash
git clone https://github.com/haorentang/NesyProAct.git
cd NesyProAct
pip install -r requirements.txt
playwright install chromium
```

LLM calls go through [LiteLLM](https://github.com/BerriAI/litellm), so any provider LiteLLM supports can be used.
Copy `.env.example` to `.env` (or export the variables) and fill in your API key:

```bash
export OPENAI_API_KEY=...            # or LITELLM_API_KEY / LITELLM_BASE_URL for a LiteLLM proxy
export LLM_MODEL_NAME=gpt-4o         # backbone, e.g. gpt-4o, gpt-4o-mini, anthropic/claude-3-5-sonnet-20240620
```

## WebArena

1. Deploy the WebArena websites following the [official instructions](https://github.com/web-arena-x/webarena/tree/main/environment_docker)
   and export their URLs:

   ```bash
   export WA_SHOPPING="http://<host>:7770"
   export WA_SHOPPING_ADMIN="http://<host>:7780/admin"
   export WA_REDDIT="http://<host>:9999"
   export WA_GITLAB="http://<host>:8023"
   export WA_MAP="http://<host>:3000"
   export WA_WIKIPEDIA="http://<host>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
   ```

2. Download WebArena's [`test.raw.json`](https://github.com/web-arena-x/webarena/tree/main/config_files) and generate per-task configs:

   ```bash
   python configs/generate_task_configs.py --raw path/to/test.raw.json
   ```

3. Run a single task or a whole site:

   ```bash
   python main.py configs/tasks/21.json
   python scripts/run_webarena.py --site shopping
   python scripts/run_webarena.py --site gitlab --task_ids 44 45 46
   ```

   Results are written to `results/<site>_<model>/<task_id>/`. Induced skills are stored in
   `skill_memory/<site>_<model>/` and reused by later tasks on the same site (set `NESY_SKILL_DIR` to change the location).
   Reset the WebArena environment between full runs.

4. Compute the success rate:

   ```bash
   python scripts/compute_success_rate.py results/shopping_gpt-4o
   ```

**Evaluation protocol.** Following ASI ([Wang et al., 2025](https://github.com/zorazrw/agent-skill-induction)), task success
is judged by an LLM evaluator (`autoeval/`, adapted from [Pan et al., 2024](https://github.com/Berkeley-NLP/Agent-Eval-Refine))
from the final page, the action history, and the agent's answer. The evaluator model defaults to `gpt-4o` at temperature 0
(`NESY_EVAL_MODEL` overrides it; `NESY_EVAL=0` disables evaluation).

## Online-Mind2Web

```bash
pip install patchright          # optional: Playwright fork with fewer automation fingerprints for live sites
python scripts/run_online_mind2web.py --tasks Online_Mind2Web.json --headless
python scripts/convert_for_webjudge.py --tasks Online_Mind2Web.json \
    --run_dir results/om2w_gpt-4o --out_dir results/om2w_gpt-4o_webjudge
```

Download the task file from the [Online-Mind2Web repository](https://github.com/OSU-NLP-Group/Online-Mind2Web) and score
the converted trajectories with its official WebJudge evaluator.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Agent backbone (LiteLLM model name) |
| `LITELLM_API_KEY`, `LITELLM_BASE_URL` | unset | Optional LiteLLM proxy credentials |
| `NESY_MAX_ACTIONS` | `3` | Maximum primitive actions per subgoal attempt |
| `NESY_REPEAT_CAP` | `0` (off) | Skip an action after N identical attempts without progress |
| `NESY_SKILL_DIR` | `skill_memory` | Where induced skills are stored |
| `NESY_EVAL` | `1` | Run the task-level LLM evaluator after each task |
| `NESY_EVAL_MODEL` | `gpt-4o` | Evaluator model |
| `NESY_JUDGE_LOG` | unset | If set, append every subgoal-judge call to this JSONL file |
| `HEADLESS` | `1` | Headless browser for `main.py` |

## Notes

- This release is a cleaned version of our research code. Few-shot examples taken from benchmark tasks and a
  site-specific navigation shortcut used during our experiments were replaced with generic, site-independent
  versions, so results can differ from the numbers reported in the paper.
- The WebArena login helper (`auto_login.py`) uses the default test accounts shipped with the WebArena Docker images.

## Citation

```bibtex
@inproceedings{tang2026nesyproact,
  title     = {{NesyProAct}: Proactive Neural-Symbolic Control for Web Agents},
  author    = {Tang, Tianyi and Xiang, Keyi and Shao, Jie-Jing and Lyu, Yueming and Tsang, Ivor and Ong, Yew-Soon and Yin, Haiyan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
