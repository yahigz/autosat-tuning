# AutoSAT Prompt Interface

## Prompt files

Place your prompt templates in this directory:

- `original_prompt.txt` — used at iteration 0 (first query to LLM)
- `feedback_prompt.txt` — used at iterations 1+ (with results from previous iteration)

Both files are Jinja2 templates. Edit them freely; the sections below are injected automatically.

---

## Available placeholders

### `{{ heuristics_section }}`

Auto-filled in both prompts. Contains a formatted block for each task being optimized, e.g.:

```
=== bump_var_function — Variable activity bumping ===
Description: This function increases the priority of variables that cause recent conflicts...
Signature: void Solver::bump_var(int var, double coeff)
Baseline code:
    if ((activity[var] += var_inc * coeff) > 1e100) { ... }
    if (vsids.inHeap(var)) vsids.update(var);
```

### `{{ baseline_section }}`

Auto-filled in `feedback_prompt.txt` only. Contains the baseline PAR-2 score and the baseline
code for the current task, e.g.:

```
=== Baseline (iteration 0) ===
PAR-2: 1234.56
Code:
void Solver::bump_var(int var, double coeff) {
    ...
}
```

### `{{ last_iter_section }}`

Auto-filled in `feedback_prompt.txt` only. Contains the best result from the previous iteration:

```
=== Last iteration result ===
PAR-2: 1100.23
Title: Adaptive decay with conflict count
Reason: Slower decay rate during early search helps stability
Code:
void Solver::bump_var(int var, double coeff) {
    ...
}
```

---

## Solver baseline markers

In `solver/baseline/EasySAT.cpp`, each optimizable region is surrounded by a pair of identical
markers on their own lines:

```cpp
void Solver::bump_var(int var, double coeff) {
<--bump_var_function-->
    if ((activity[var] += var_inc * coeff) > 1e100) { ... }
    if (vsids.inHeap(var)) vsids.update(var);
<--bump_var_function-->
}
```

The marker name (e.g. `bump_var_function`) becomes the **task name** that the optimizer uses.

**Rules for markers:**
- The marker must appear on its own line: `<--task_name-->`
- Use the same marker name for the opening and closing line
- The code between the two markers is the baseline for that task
- If the LLM does not return code for a task, the baseline code is used unchanged

### Supported tasks (defined in `solver/baseline/EasySAT.cpp`)

| Task name           | Region                                      |
|---------------------|---------------------------------------------|
| `bump_var_function` | Body of `void Solver::bump_var(...)`        |
| `restart_function`  | Body of `void Solver::restart()`            |
| `rephase_function`  | Body of `void Solver::rephase()`            |
| `restart_condition` | `else if (...) restart();` line in `solve()`|
| `rephase_condition` | `else if (...) rephase();` line in `solve()`|

To add a new task: wrap the relevant code region with `<--my_task-->` markers and add its
metadata to `heuristic_modules` in `config.yaml`.

---

---

## task_selection_mode

| Mode         | Behaviour |
|--------------|-----------|
| `random_one` | One random task is picked per iteration; LLM returns one function |
| `cycle`      | Tasks are cycled in order; LLM returns one function per iteration |
| `all`        | **All tasks in one call**: LLM returns code for every heuristic at once |

### `all` mode — structured output format

When `use_structured: true` the model must return:
```json
{
  "implementations": [
    {"task": "bump_var_function", "code": "...", "title": "...", "reason": "..."},
    {"task": "restart_function",  "code": "...", "title": "...", "reason": "..."},
    ...
  ]
}
```
Include one entry for every task in `optimize_tasks`. If no improvement found for a task,
include the baseline code unchanged.

### `all` mode — plain text fallback (use_structured: false)

Wrap each function with task-specific markers:
```
// start_bump_var_function
void Solver::bump_var(int var, double coeff) {
    ...
}
// end_bump_var_function

// start_restart_function
void Solver::restart() {
    ...
}
// end_restart_function
```

---

## Example config.yaml

```yaml
iteration_num: 10
batch_size: 4
data_parallel_size: 2
timeout: 2
data_dir: "./temp/data_train"

optimize_tasks:
  - bump_var_function
  - restart_function

task_selection_mode: random_one   # random_one | cycle

original: true   # run original solver to get baseline PAR-2

llm_model: "claude-sonnet-4-6"

eval_timeout: 5
eval_data_dir: "./temp/data_eval"
eval_parallel_size: 1

template_update_strategy: none   # none | greedy_train | annealing
```
