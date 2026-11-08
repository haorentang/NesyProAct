"""Prompts for the transition, state, and action layers, the subgoal judges, and skill induction/reuse.

All few-shot examples use a generic public-library catalog website that is not part of any evaluation benchmark.
"""

# ---------------------------------------------------------------------------
# Transition layer: action-query program for the current page
# ---------------------------------------------------------------------------
ACTION_PLAN_PROMPT = """You are a UI environment action predictor. Your task is to predict the required actions that can be performed on the current web page in order to reach a goal.

You are given:
- The overall goal of the task
- The subtask to be executed
- The already executed action history of the current subtask
- The current web page represented as an accessibility tree with every element whose ID, role, and value are specified.

Instructions:
1. You should output a sequence of action objects arranged in execution order **scoped by the current page**, which means a page transition or an information extraction (using 'store' or 'send_msg_to_user') should be expected after the actions have been executed.
2. One action object consists of the action type, the action intent label, and the action hint:
- Available action types: click(bid), fill(bid, text), hover(bid), press(bid, 'Enter'), select_option(bid, option), go_back(), go_forward(), send_msg_to_user(text), store(text), noop()
    - Special action type store(text): To store a piece of information for later use, the storage will be read before every send_msg_to_user(text) action
    - **Use store(text) to store intermediate information, and use send_msg_to_user(text) for information that the overall goal requires.**
- The action intent label is a single word describing the intent of the action. A mapping between action types and their corresponding intent labels is provided. You can only use the action types that are allowed for the intent label.
{INTENT_LABELS}
- The action hint is the expected result of the action in descriptive language that serves as a hint for the grounding agent to find the target element id(bid) for the action.
3. **Important**: Output the action sequence in the YAML format surrounded by ```yaml and ```:
```yaml
- action: <action_type>
  intent: <intent_label>
  hint: <action_hint>
```
===
Examples (a public library catalog website):
<example_1>
Overall goal: Find the newest book about machine learning
Subtask: Search for machine learning books
Current page: [library catalog homepage with a search box]

Response:
The page has a search box. "fill" and "press" actions can be used to enter the keywords and submit the search, then the page should transition to a results page.
```yaml
- action: fill
  intent: Search
  hint: Type "machine learning" in the catalog search box
- action: press
  intent: Search
  hint: Press Enter to submit the search
```
</example_1>

<example_2>
Overall goal: Find the newest book about machine learning
Subtask: Sort results by publication date
Current page: [search results page with a "Sort by" dropdown offering "Relevance" and "Publication date (newest first)"]

Response:
The page has a "Sort by" dropdown with a publication-date option. "select_option" can be used to reorder the results, then the page should display the results in a different order.
```yaml
- action: select_option
  intent: Sort
  hint: Select the "Publication date (newest first)" option from the "Sort by" dropdown
```
</example_2>

<example_3>
Overall goal: Find the newest book about machine learning
Subtask: List the title of the newest book
Current page: [search results sorted by publication date, the first item titled "Learning Systems in Practice"]

Response:
The results are sorted with the newest item first and its title is visible. "send_msg_to_user" can be used to report the title, completing the information request.
```yaml
- action: send_msg_to_user
  intent: Extract
  hint: Report the title of the first (newest) result
```
</example_3>
"""

ACTION_PLAN_USER_PROMPT = """Overall goal: {overall_goal}
Subtask: {subtask}
Current page:
{current_ax_tree}
The already executed action history of the current subtask: {action_history}
===
Response:
"""

# ---------------------------------------------------------------------------
# State / action layers: grounding an action to a page element
# ---------------------------------------------------------------------------
GROUNDING_PROMPT = """You are a UI grounding expert. Your task is to find the target element for an action in the process of completing a subtask.

You are given:
- The overall task goal
- The current subtask
- The action type
- The expected result of the action in descriptive language that serves as a hint.
- A few candidate elements with their IDs, roles, values, and other attributes.

Instructions:
- You should first contemplate what result may be induced if the action is taken on the candidate elements.
- Output a rationale for the thinking process and the action calling function string surrounded by <>.
- The action must include the bid of the target element.
- Output format:
[rationale]
<[action_calling_function_string, e.g. click('15'), fill('15', 'text'), hover('15'), select_option('15', 'option')]>

Examples:
The hint indicates we need to open the account menu. Element "123" is a generic button with value "Help", element "456" is a navigation link "Events", and element "789" is a button with value "My account". Element "789" matches this intent.
<click('789')>

The hint indicates we need to type "machine learning" into the catalog search box. Element "512" is a searchbox with the placeholder "Search the catalog". It matches this intent.
<fill('512', 'machine learning')>

The hint indicates we need to press Enter to submit the search. Element "512" is the searchbox that now contains "machine learning". It matches this intent.
<press('512', 'Enter')>

The hint indicates we need to order results by publication date. Element "543" is a combobox labelled "Sort by" with an option "Publication date (newest first)". It matches this intent.
<select_option('543', 'Publication date (newest first)')>
"""

GROUNDING_USER_PROMPT = """The overall task goal: {task_goal}
The current subtask: {subtask}
The action type out of click, fill, hover, and press: {action_type}
The expected result of the action in descriptive language that serves as a hint: {hint}
A few candidate elements with their IDs, roles, values, and other attributes represented as strings:
{candidate_elements}
===
"""

RAW_GROUNDING_PROMPT = """You are a UI grounding expert. Your task is to find the target element for an action in the process of completing a subtask.

You are given:
- The overall task goal
- The current subtask
- The action type out of click('bid'), fill('bid', 'text'), hover('bid'), press('bid', 'Enter'), and select_option('bid', 'option')
- The expected result of the action in descriptive language that serves as a hint.
- The current web page represented as an accessibility tree with every element whose ID, role, and value are specified.

Instructions:
- Output a rationale for the thinking process and the action calling function string surrounded by <>.
- Output format:
[rationale]
<[action_calling_function_string, e.g. click('15'), fill('15', 'text'), hover('15')]>

Examples:
The hint indicates we need to open the account menu. Element "789" is a button with value "My account", which matches this intent.
<click('789')>

The hint indicates we need to type "machine learning" into the catalog search box. Element "512" is a searchbox with the placeholder "Search the catalog". It matches this intent.
<fill('512', 'machine learning')>

The hint indicates we need to order results by publication date. Element "543" is a combobox labelled "Sort by" with an option "Publication date (newest first)". It matches this intent.
<select_option('543', 'Publication date (newest first)')>
"""

RAW_GROUNDING_USER_PROMPT = """The overall task goal: {task_goal}
The current subtask: {subtask}
The action type out of click, fill, hover, press, and select_option: {action_type}
The expected result of the action in descriptive language that serves as a hint: {hint}
===
Accessibility Tree of the current web page:
{ax_tree}
"""

TEXT_GROUNDING_PROMPT = """You are a UI grounding expert. Your task is to find the target message to send in a web page for finishing the subtask.

You are given:
- The overall task goal
- The current subtask
- Some extra information
- The current web page represented as an accessibility tree with every element whose ID, role, and value are specified.

Instructions:
- The subtask or the overall task goal is asking for specific information possibly hidden in the web page.
- Output a rationale for the thinking process and the target message surrounded by <>.
- Output format:
[rationale]
<[target_message]>

Example 1:
Task goal: Find the opening hours of the central branch on Saturday
Subtask: Extract the Saturday opening hours

Rationale: The page lists opening hours per weekday. The row labelled "Saturday" contains the text "10:00 - 16:00", which is the requested information.
<'The central branch is open 10:00 - 16:00 on Saturday'>

Example 2:
Task goal: Find how many copies of "Learning Systems in Practice" are available
Subtask: Report the number of available copies
Extra information:
- Two branches were checked before this page

Rationale: The availability table on the current page shows "Available: 3" for this title, which answers the question.
<'3 copies of "Learning Systems in Practice" are available'>
"""

TEXT_GROUNDING_USER_PROMPT = """The overall task goal: {task_goal}
The current subtask: {subtask}
Extra information:
{extra_info}
===
Accessibility Tree of the current web page:
{ax_tree}
"""

# ---------------------------------------------------------------------------
# Subgoal judges
# ---------------------------------------------------------------------------
# Used after synthesizing and executing a new action-query program (SkillManager).
SUBGOAL_JUDGE_PROMPT = """You are a task execution control expert that can determine if a subtask is completed.

You are given:
- The overall goal of the task
- The subtask to be executed
- The current web page represented as an accessibility tree with every element whose ID, role, and value are specified.

Instructions:
- Determine whether the subtask has completed after the current page is reached
- The current web page can be an intermediate page during the execution of the subtask, or the goal web page
- Output a rationale identifying key components of the page, and an indicator out of "<completed>" or "<not_completed>"

Examples (a public library catalog website):
<example_1>
Overall goal: Find the newest book about machine learning
Subtask: Sort results by publication date
Current page: [search results for "machine learning" sorted by relevance]

Response:
The page shows search results ordered by relevance. The subtask requires the results to be ordered by publication date, which is not yet the case.
<not_completed>
</example_1>

<example_2>
Overall goal: Renew a borrowed book
Subtask: Navigate to the account page
Current page: [account page with the heading "My account" and a "Current loans" table]

Response:
The page header reads "My account" and lists the current loans, so the account page has been reached.
<completed>
</example_2>
"""

# Used when a stored skill is replayed (SkillExecutor reuse path).
SKILL_REPLAY_JUDGE_PROMPT = """You are a web task judge that can judge whether a subgoal has been reached by inspecting the resulting web page.

You are given:
- The executed subgoal
- The resulting web page represented as an accessibility tree with every element whose ID, role, and value are specified.
- The executed action history of the current subtask.

Instructions:
- First focus on some key elements that may be indicative of the subgoal being reached.
- If the subgoal modifies the current page (e.g., filling or submitting a form), also use the action history to judge whether the modification was performed.
- Output a rationale for the thinking process and your judgement surrounded by <>.

Below are the judgement candidates, only one of them should be selected:
<success>
<failed>
<incomplete>

Example:
Executed subgoal: Search for machine learning books

Rationale: The page now shows a results list with items such as "Learning Systems in Practice" and "Introduction to Statistical Learning", so the search results are displayed.
<success>
"""

JUDGE_USER_PROMPT = """The executed subgoal: {executed_subgoal}
The resulting web page represented as an accessibility tree with every element whose ID, role, and value are specified:
{ax_tree}
The executed action history of the current subtask:
{action_history}

Response:
"""

# ---------------------------------------------------------------------------
# Skill induction
# ---------------------------------------------------------------------------
LLM_JUDGE_SUBTASK_PROMPT = """You are a skill extraction expert that generates generalized skill representations from subtask executions.

You are given:
- The subtask query
- The execution steps that were performed
- The final page state (URL, errors, etc.)
- The current page Accessibility Tree

Your task:
Generate a generalized skill representation:
- A skill name that abstracts the specific task (e.g., 'search_item' for 'search for machine learning books')
- A one-sentence description of what this skill does.

The skill should be reusable if and only if:
- Execution finished without errors,
- The skill represents a common web interaction pattern.
- When the skill is reused, it will be applied to cases with different entities, but EXACTLY the same action trajectory (SAME BUTTON) is performed.
- If re-execution would require adapting the action sequence based on the current page content/state, do NOT generate a reusable skill and set skill_name to null.
- **Critical**: If the skill is not reusable or you are unsure about the reusability, set skill_name to null.

Skill naming rules:
- Use the format: <verb>_<object>[_<fixed_intent>].
- The verb must be the same as the query verb.
- If the functional objective is fixed, include it as a suffix (e.g., sort_results_by_date, navigate_to_account).
- Do NOT include page-specific identifiers, DOM paths, or element indices.
- Only entity values may vary across executions; intent objectives must remain fixed in the skill name
  (e.g., sort_results_by_date rather than sort_results), and must not generalize over different intent objectives.

Skill description rules:
- Write one or two sentences.
- Describe the semantic interaction procedure, not the final outcome.
- Do NOT reference DOM structure, element IDs, or absolute positions.
- The skill description must remain intent-specific and explicitly encode the fixed functional objective.
- Use action-oriented phrasing: "Open...", "Click...", "Enter...", "Select..." to describe the sequence.

Examples:
- Query: 'search for machine learning books' -> Skill name: 'search_item', Description: 'Enter the item name in the search box and execute the search'
- Query: 'fill email field with jane@example.com' -> Skill name: 'fill_text_field', Description: 'Fill a text input field with the specified value'

Respond with JSON:
{{
    "skill_name": "generalized skill name (e.g., 'search_item', 'sort_results_by_date'). Use null if the skill cannot be reused.",
    "skill_description": "one-or-two-sentence description of the skill. Use null only if the skill cannot be determined at all."
}}
"""

LLM_JUDGE_SUBTASK_USER_PROMPT = """Subtask Query: {query}
Number of steps executed: {num_steps}
Steps executed:
{steps_summary}
Final URL: {url_after}
Execution error: {error}
Execution succeeded (no error): {execution_success}

{ax_tree_section}

Evaluate the execution and generate the skill representation as specified.
"""

# ---------------------------------------------------------------------------
# Skill reuse
# ---------------------------------------------------------------------------
SKILL_REUSE_PROMPT = """You are a skill reuse expert who determines whether a skill can be reused to complete the current subtask.

You are given:
- The overall task goal.
- The query of the current subtask.
- All available skills in the current situation, their descriptions, and parameters (if any).

You will return:
- Whether a skill can be reused, the skill name and parameter if any.
**Critical**: Return only the EXACT skill name if it can be reused and null if no skill can be reused.

Instructions:
- If a skill can be reused, the skill name and description should match EXACTLY with the intention of the subgoal.
- If a skill has parameters, fill in the parameter based on the given query and task goal.
- If a skill does not require parameters, set the parameter to null.
- If no skill can be reused, set the name to null and parameter to null.

Examples:
Subtask goal: Search for machine learning books
Skills available: search_item(parameter required: True), sort_results_by_date(parameter required: False)
Response: {"name":"search_item", "parameter":"machine learning"}

Subtask goal: Navigate to the account page
Skills available: navigate_to_account(parameter required: False), navigate_to_events(parameter required: False)
Response: {"name":"navigate_to_account", "parameter":null}

Subtask goal: Navigate to the account page
Skills available: navigate_to_events(parameter required: False)
Response: {"name":null, "parameter":null}

Output format strictly in JSON format:
{
    "name":"skill_name",
    "parameter":"parameter"
}
"""

SKILL_REUSE_USER_PROMPT = """
Task goal: {task_goal}
Subtask goal: {subtask_goal}
===
Skills available: {skills_avaliable}
"""
