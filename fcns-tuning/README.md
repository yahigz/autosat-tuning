# FCNS Tuning

LLM-driven optimizer for the FCNS graph coloring solver. The pipeline iteratively edits marker-based C++ heuristic blocks, runs the solver on the training graphs, captures solver traces from `stderr`, and ranks candidates by two metrics:

- number of colors
- solver time to reach that coloring

The same run also produces plots and a live monitoring dashboard.

## What the pipeline does

- Loads marker-based heuristic tasks from `fcns-tuning/solver/baseline/fcns.cpp`
- Builds prompts from the project templates in `fcns-tuning/prompts/`
- Queries the configured LLM for candidate heuristic edits
- Compiles and evaluates each candidate on the training set
- Captures trace events from `stderr` with timestamps
- Stores results, checkpoints, plots, and progress snapshots under `fcns-tuning/results/runs/<run_id>/`
- Optionally starts a local HTTP dashboard for live monitoring

## Quick Start

```bash
cd fcns-tuning
uv run python main.py --config config.yaml
```

If you want to use a different config file, pass it with `--config`.

## Monitoring Dashboard

Enable the dashboard in `config.yaml`:

```yaml
enable_server: true
server_port: 8080
```

When enabled, the runner starts a local server and writes the current run state to `results/runs/<run_id>/progress.json`.
Open the dashboard in a browser at:

```text
http://localhost:8080
```

The dashboard shows:

- a primary metric chart for colors
- a secondary metric chart for time
- a per-iteration table with the current best candidate
- a code diff viewer for comparing two selected iterations

The server also remains backward-compatible with the older autosat-style `baseline_par2` payload, but FCNS runs publish `metric_mode: colors_time` and `metric_labels` for the FCNS metrics.

## Input Format

The runner accepts both binary `.col.b` instances and plain-text graph files.

### Binary `.col.b`

These files contain a `p edge n m` header followed by a binary upper-triangle adjacency bitmap. The loader:

- reads the vertex count `n` from the header
- decodes the bitmap into edges `(u, v)` with `u < v`
- removes duplicates and self-loops
- converts the graph into the stdin format expected by the solver

### Plain-text graph files

Plain-text files are parsed as:

- first line: a graph header containing `n` and `m`
- remaining lines: edge endpoints

The loader extracts integer pairs from the file, normalizes edges, and writes the solver input as:

```text
n m
u1 v1
u2 v2
...
```

Vertices are treated as 0-based when validating the graph instance.

## Prompts and Structured Output

Project-specific prompt defaults live in `fcns-tuning/prompting.py`.

The shared prompt helpers in `autosat_core/prompting.py` now support:

- project-specific default heuristic module names
- custom baseline metadata via `baseline_info`
- configurable structured-output fields

For FCNS, the structured response is expected to describe the edited heuristic blocks for the current marker tasks. The exact marker names are user-defined, so the prompt defaults should match the names you put into `solver/baseline/fcns.cpp`.

## Results Layout

A typical run writes:

```text
results/runs/<run_id>/
  baseline_result.json
  final_result.json
  progress.json
  prompts/
  checkpoints/
  iterations/
  plots/
```

Each iteration stores the candidate source, summary JSON, and per-instance trace data. The plotting code also writes fixed-scale charts for each train instance and an aggregate 3D iteration plot.

## Configuration

Important config keys in `config.yaml`:

- `iteration_num` - number of optimization iterations
- `batch_size` - number of LLM candidates per iteration
- `task_selection_mode` - `all`, `cycle`, or `random_one`
- `timeout` - per-instance solver timeout in seconds
- `run_eval` - run the evaluation set after training
- `resume_from_checkpoint` - resume from the latest checkpoint if available
- `enable_server` - start the live dashboard
- `server_port` - dashboard port
- `use_structured` - request structured LLM output

## Notes

- The solver baseline is intentionally left marker-based so you can choose the marker names yourself.
- Put each marker on its own line as a comment, for example `// <--uvertex_function-->`. The adapter recognizes the marker line and strips the whole marked block before compiling the generated solver.
- Progress is written as JSON and the dashboard reads it continuously.
- The plots use one fixed scale across iterations so that color/time changes stay comparable.
