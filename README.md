# AutoSAT Tuning

LLM-driven optimizer for SAT solver heuristics. Iteratively proposes C++ improvements,
compiles and benchmarks them on SAT instances, then feeds results back to the model.

Reusable core code now lives in `autosat_core/`, so a different solver can reuse
the prompting, execution, evaluation, plotting, and source-rendering helpers.

### Library API

- Define optimization functions as `TaskSpec` objects from `autosat_core.tasks`.
- Use `MarkerSolverAdapter` from `autosat_core.marker_adapter` for marker-based C++ solvers.
- Build prompts with `build_prompt_text_for_tasks(...)`; the response schema stays fixed as `code`, `title`, `reason`.
- Provide any solver-specific metadata or prompt text through task fields and config payloads.

Example:

```python
from pathlib import Path
from autosat_core import MarkerSolverAdapter, TaskSpec, build_prompt_text_for_tasks

adapter = MarkerSolverAdapter(
    name="MySolver",
    baseline_cpp=Path("solver/baseline/MySolver.cpp"),
    task_specs=[
        TaskSpec(name="restart_function", label="Restart policy"),
        TaskSpec(name="rephase_function", label="Rephase policy"),
    ],
)
prompt = build_prompt_text_for_tasks(
    base_prompt_text=Path("prompts/original_prompt.txt").read_text(),
    task_specs=adapter.task_specs,
    baseline_codes={t.name: adapter.extract_baseline_section(t.name) for t in adapter.task_specs},
)
```

---

## Directory layout

```
autosat-tuning/
├── autosat_core/             # reusable library for any solver
├── main.py                    # entry point
├── config.yaml                # main config (edit this)
├── pyproject.toml             # Python deps (managed by uv)
│
├── prompts/
│   ├── original_prompt.txt    # first-iteration prompt  ← you write this
│   ├── feedback_prompt.txt    # feedback prompt         ← you write this
│   └── INTERFACE.md           # placeholder reference
│
├── solver/
│   ├── baseline/
│   │   ├── EasySAT.cpp        # original solver WITH <--markers-->
│   │   ├── EasySAT.hpp
│   │   └── heap.hpp
│   └── template/              # written by the optimizer (best known state)
│
├── prompting.py               # prompt building (section injection)
├── server.py                  # HTTP dashboard with diff viewer
├── utils.py                   # result collection, file helpers
├── plotting.py                # training curve PNG
│
├── llm_api/base_api.py        # LLM backends (GPT, Eliza/Claude, local)
├── execution/execution_worker.py
└── evaluation/evaluate.py
```

**Runtime outputs** (created automatically):

```
temp/results/               solver stdout files
temp/runs/<run_id>/         working solver copies for compilation
results/runs/<run_id>/
  baseline_result.json
  final_result.json
  eval_best_result.json     ← best config found, written at eval stage
  progress.json             ← consumed by HTTP dashboard
  checkpoints/
  snapshots/
  eval_results/
```

---

## Quickstart

```bash
# 1. Install deps (once)
uv sync

# 2. Put SAT instances into training and eval directories
mkdir -p temp/data_train temp/data_eval
cp your_cnf_files/*.cnf temp/data_train/
cp your_cnf_files/*.cnf.xz temp/data_train/  # optional
cp your_eval_cnf_files/*.cnf temp/data_eval/
cp your_eval_cnf_files/*.cnf.xz temp/data_eval/  # optional

# 3. Set API credentials
cp .env.example .env      # fill in AUTOSAT_API_KEY etc.

# 4. Run
uv run python main.py --config config.yaml
```

---

## Markers in `solver/baseline/EasySAT.cpp`

Each optimizable region is fenced by a pair of identical marker lines:

```cpp
void Solver::bump_var(int var, double coeff) {
<--bump_var_function-->
    if ((activity[var] += var_inc * coeff) > 1e100) {
        for (int i = 1; i <= vars; i++) activity[i] *= 1e-100;
        var_inc *= 1e-100;
    }
    if (vsids.inHeap(var)) vsids.update(var);
<--bump_var_function-->
}
```

**Rules:**
- Each marker must be on its own line: `<--task_name-->`
- Opening and closing markers are identical
- The code between the markers is the **baseline** for that task
- The marker name becomes the task name (e.g. `bump_var_function`)
- If the LLM does not return code for a task, the baseline code is used unchanged

Supported tasks out of the box:

| Task name            | What it controls                          |
|----------------------|-------------------------------------------|
| `bump_var_function`  | Body of `void Solver::bump_var(int, double)` |
| `restart_function`   | Body of `void Solver::restart()`          |
| `rephase_function`   | Body of `void Solver::rephase()`          |
| `restart_condition`  | `else if (...) restart();` in `solve()`   |
| `rephase_condition`  | `else if (...) rephase();` in `solve()`   |

To add a new task: wrap a code region with markers, add metadata under
`heuristic_modules:` in `config.yaml`.

The solver accepts both plain `.cnf` and compressed `.cnf.xz` instances.

---

## Prompt files (`prompts/`)

Both files are **Jinja2 templates**. You write all the static content;
three placeholders are injected automatically:

| Placeholder                 | Available in           | Injected content |
|-----------------------------|------------------------|------------------|
| `{{ heuristics_section }}`  | both prompts           | Task list with description, signature, baseline code |
| `{{ baseline_section }}`    | `feedback_prompt.txt`  | Baseline PAR-2 + baseline code for current task |
| `{{ last_iter_section }}`   | `feedback_prompt.txt`  | Best result from previous iteration: PAR-2, code, title, reason |

### `prompts/original_prompt.txt` (example)

```
You are a SAT solver researcher optimizing heuristics in EasySAT.

{{ heuristics_section }}

Write an improved implementation. Wrap your code as:
// start
<implementation here>
// end

Tips:
1) Valid C++ only, no pseudocode or markdown fences inside the code.
2) Self-check: balanced braces, semicolons, valid declarations.
3) Must differ from the baseline.
```

### `prompts/feedback_prompt.txt` (example)

```
You are a SAT solver researcher optimizing heuristics in EasySAT.

{{ heuristics_section }}

{{ baseline_section }}

{{ last_iter_section }}

Based on the above, write an improved implementation. Wrap as:
// start
<implementation here>
// end

Tips:
1) Valid C++ only.
2) Fix any syntax errors from the previous attempt.
3) Must differ from the baseline and the previous attempt.
```

---

## All config parameters

```yaml
# ── Training ──────────────────────────────────────────────────────────────────
iteration_num: 10           # total optimization iterations
batch_size: 4               # LLM queries per iteration (candidates per round)
data_parallel_size: 2       # parallel solver processes per candidate
devoid_duplication: false   # skip candidate if identical code was already tried
timeout: 2                  # per-instance time limit in seconds (train)
data_dir: "./temp/data_train"

# ── Tasks ─────────────────────────────────────────────────────────────────────
optimize_tasks:             # marker names to optimize; order matters for cycle mode
  - bump_var_function
  - restart_function
task_selection_mode: random_one   # random_one | cycle
rand_seed: 42                     # RNG seed for task selection in random_one mode

# ── Baseline ──────────────────────────────────────────────────────────────────
original: true              # true  → compile and run original solver for baseline PAR-2
                            # false → use original_result values below
original_result:            # only used when original: false
  time: 0
  PAR-2: 0

# ── LLM ───────────────────────────────────────────────────────────────────────
llm_model: "claude-sonnet-4-6"
api_base: ""                # OpenAI-compatible base URL; leave empty for Eliza
api_key:  ""                # overridden by AUTOSAT_API_KEY env var
use_structured: true        # request JSON output with code / title / reason fields
temperature: 1.0            # LLM sampling temperature

# ── Custom heuristic module metadata ──────────────────────────────────────────
# Override defaults from prompting.py (all fields optional)
heuristic_modules:
  bump_var_function:
    label: "Variable activity bumping"
    description: "..."
    signature: "void Solver::bump_var(int var, double coeff)"
    insertion_format: "Replace the full function body."

# ── Template update during training ───────────────────────────────────────────
template_update_strategy: none
# none          – never update solver/template/ during training (default)
# greedy_train  – update solver/template/ when train PAR-2 strictly improves
# annealing     – SA Metropolis acceptance (temperature log-anneals 1.0 → 0.1)

# ── Evaluation ────────────────────────────────────────────────────────────────
run_eval: true              # run eval stage after training finishes
eval_timeout: 5             # per-instance time limit for evaluation
eval_data_dir: "./temp/data_eval"
eval_parallel_size: 1
keep_intermediate_results: false

# ── Checkpointing / resume ────────────────────────────────────────────────────
resume_from_checkpoint: true
run_id: ""                  # leave empty to start a new run;
                            # set to a specific run_id to resume it

# ── Eval-only mode ────────────────────────────────────────────────────────────
eval_only_from_run: false   # skip training, run eval stage on an existing run

# ── HTTP dashboard ────────────────────────────────────────────────────────────
enable_server: false        # true → start dashboard
server_port: 8080
```

---

## CLI parameters

All config keys can also be passed as CLI flags (flags override config file):

```
uv run python main.py [OPTIONS]

  --config PATH                 YAML config file               [./config.yaml]
  --iteration_num INT           Total training iterations
  --batch_size INT              LLM queries per iteration
  --data_parallel_size INT      Parallel solver processes per candidate
  --devoid_duplication BOOL     Skip duplicate code
  --timeout INT                 Train time limit per instance (s)
  --data_dir PATH               Training CNF directory
  --optimize_tasks STR [STR...] Task marker names
  --task_selection_mode STR     random_one | cycle
  --rand_seed INT               RNG seed
  --original BOOL               Run original solver for baseline
  --original_result DICT        Baseline result when original=false
  --llm_model STR               LLM model name
  --api_base STR                OpenAI-compatible base URL
  --api_key STR                 API key
  --use_structured BOOL         Structured JSON output
  --temperature FLOAT           LLM sampling temperature
  --template_update_strategy STR  none | greedy_train | annealing
  --run_eval BOOL               Run eval stage after training
  --eval_timeout INT            Eval time limit per instance (s)
  --eval_data_dir PATH          Eval CNF directory
  --eval_parallel_size INT      Parallel eval processes
  --keep_intermediate_results BOOL  Keep temp files
  --resume_from_checkpoint BOOL Resume from checkpoint
  --run_id STR                  Specific run ID to resume
  --eval_only_from_run BOOL     Skip training, eval existing run
  --enable_server BOOL          Start HTTP dashboard
  --server_port INT             Dashboard port                  [8080]
```

---

## LLM backends

| Backend            | `AUTOSAT_API_TYPE` | Trigger                                         |
|--------------------|--------------------|-------------------------------------------------|
| Eliza (Claude)     | `eliza`            | Set `AUTOSAT_API_TYPE=eliza` in `.env`          |
| OpenAI-compatible  | (unset)            | `api_base` is non-empty, or model is `gpt-*`    |
| Local / vLLM       | (unset)            | `llm_model` is `Qwen` / `llama` / `deepseek`   |

### Eliza (internal Yandex)

```bash
# .env
AUTOSAT_API_TYPE=eliza
AUTOSAT_API_KEY=your_soy_oauth_token
AUTOSAT_LLM_MODEL=claude-sonnet-4-6   # optional override
```

Available models: `claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4-5-20251001`

### OpenAI / DeepInfra

```bash
# .env
AUTOSAT_API_BASE=https://api.deepinfra.com/v1/openai
AUTOSAT_API_KEY=your_key
AUTOSAT_LLM_MODEL=meta-llama/Meta-Llama-3-70B-Instruct
```

---

## Dummy configs

Save these as separate YAML files and run with `--config <file>`.

### `config.mini.yaml` — fast smoke test

```yaml
iteration_num: 2
batch_size: 2
data_parallel_size: 1
timeout: 1
data_dir: "./temp/data_train"
optimize_tasks: [bump_var_function]
task_selection_mode: random_one
original: true
llm_model: "claude-sonnet-4-6"
use_structured: true
run_eval: false
enable_server: false
```

### `config.eval_only.yaml` — re-run eval on existing run

```yaml
eval_only_from_run: true
run_id: "run_20250618_120000_12345_abcd"   # paste your run_id here
eval_timeout: 5
eval_data_dir: "./temp/data_eval"
eval_parallel_size: 2
keep_intermediate_results: false
llm_model: "claude-sonnet-4-6"
```

### `config.multitask.yaml` — all tasks, cycling, with dashboard

```yaml
iteration_num: 20
batch_size: 4
data_parallel_size: 2
timeout: 2
data_dir: "./temp/data_train"
optimize_tasks:
  - bump_var_function
  - restart_function
  - rephase_function
  - restart_condition
  - rephase_condition
task_selection_mode: cycle
template_update_strategy: greedy_train
original: true
llm_model: "claude-sonnet-4-6"
use_structured: true
run_eval: true
eval_timeout: 10
eval_data_dir: "./temp/data_eval"
eval_parallel_size: 4
enable_server: true
server_port: 8080
```

### `config.local_llm.yaml` — local vLLM / DeepInfra, no structured output

```yaml
iteration_num: 10
batch_size: 4
data_parallel_size: 2
timeout: 2
data_dir: "./temp/data_train"
optimize_tasks: [bump_var_function]
task_selection_mode: random_one
original: true
# api_base and api_key from .env:
# AUTOSAT_API_BASE=https://api.deepinfra.com/v1/openai
# AUTOSAT_API_KEY=your_key
llm_model: "meta-llama/Meta-Llama-3-70B-Instruct"
use_structured: false   # plain text for models without json_schema
run_eval: true
eval_timeout: 5
eval_data_dir: "./temp/data_eval"
eval_parallel_size: 1
enable_server: false
```

---

## HTTP Dashboard

Enable with `enable_server: true` in config, then open `http://localhost:8080`.

**PAR-2 chart** — live graph of best PAR-2 per iteration vs baseline (updates every 5 s).

**Iteration table** — one row per iteration showing task, PAR-2, delta, title, reason, code preview.

**Code diff** — click any row to mark it **A** (green border), click another to mark **B** (blue border).
A side-by-side diff appears below the table with line-level highlighting:
green lines added, red lines removed, grey lines unchanged.
Clicking a selected row again deselects it.

---

## Environment variables

| Variable                    | Description                                | Default      |
|-----------------------------|--------------------------------------------|--------------|
| `AUTOSAT_API_TYPE`          | `eliza` or empty (inferred)                | inferred     |
| `AUTOSAT_LLM_MODEL`         | Model name (overrides config)              | from config  |
| `AUTOSAT_API_BASE`          | API base URL (overrides config)            | from config  |
| `AUTOSAT_API_KEY`           | API key (overrides config `api_key`)       | —            |
| `AUTOSAT_STRUCTURED_OUTPUT` | `1` = enabled / `0` = disabled             | `1`          |
| `AUTOSAT_API_RETRY_SECONDS` | Seconds between retries on API error       | `10`         |
| `AUTOSAT_API_MAX_RETRIES`   | Max retries (0 = infinite)                 | `0`          |

Place these in `.env` at the project root (loaded automatically at startup).

---

## `.env.example`

```bash
# Copy to .env and fill in your values

# --- Eliza (Yandex internal) ---
AUTOSAT_API_TYPE=eliza
AUTOSAT_API_KEY=your_soy_oauth_token_here
AUTOSAT_LLM_MODEL=claude-sonnet-4-6

# --- OR: OpenAI / DeepInfra / vLLM ---
# AUTOSAT_API_BASE=https://api.deepinfra.com/v1/openai
# AUTOSAT_API_KEY=your_api_key_here
# AUTOSAT_LLM_MODEL=meta-llama/Meta-Llama-3-70B-Instruct

# --- Optional ---
# AUTOSAT_STRUCTURED_OUTPUT=1
# AUTOSAT_API_RETRY_SECONDS=10
# AUTOSAT_API_MAX_RETRIES=3
```
