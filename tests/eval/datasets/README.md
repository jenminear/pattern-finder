# Evaluation Datasets

This directory contains evaluation datasets for testing agent behavior.

## Running Evaluations

### Default Dataset

`basic-dataset.json`'s `exact_pattern_match_applies_correctly` and
`insufficient_data_stays_honest` cases depend on DB state that isn't there
on a fresh checkout -- seed it once first (idempotent, safe to rerun):

```bash
uv run python tests/eval/seed_eval_data.py
```

**`agents-cli eval generate` doesn't work with this agent as of CLI 0.5.0** --
it delegates to Vertex AI's managed inference (`vertexai._genai.evals.
run_inference`), which builds its `AgentConfig` by calling
`inspect.signature()` on every item in `tools`. That fails for `McpToolset`
(a toolset, not a single callable) with "McpToolset object is not a
callable object" -- confirmed, not a config mistake on our end. Worth
re-checking against a newer CLI version later. Until then, generate traces
locally through ADK's own Runner instead (same trace schema `eval grade`
expects, just produced without the broken managed-inference step):

```bash
uv run python tests/eval/generate_traces_local.py
agents-cli eval grade --traces artifacts/traces/local_traces.json --config tests/eval/eval_config.yaml
```

### Custom Dataset
```bash
# Generate traces for a custom dataset
agents-cli eval generate --dataset tests/eval/datasets/custom-dataset.json --output custom_traces/
agents-cli eval grade --metrics general_quality --traces custom_traces/
```

## Dataset Format

Each dataset file follows the Gemini Enterprise Agent Platform Evaluation
dataset format. An eval case may use **either** of two shapes — both are
valid input to `agents-cli eval generate`:

**Shape A — single-prompt case:**

```json
{
  "eval_cases": [
    {
      "eval_case_id": "unique_case_id",
      "prompt": {
        "role": "user",
        "parts": [{"text": "User message"}]
      }
    }
  ]
}
```

**Shape B — continued-conversation case (the "N+1" pattern):**
The case carries prior turns in `agent_data` and the last turn ends with a
user message; `eval generate` appends the next agent response.

```json
{
  "eval_cases": [
    {
      "eval_case_id": "unique_case_id",
      "agent_data": {
        "turns": [
          {
            "turn_index": 0,
            "events": [
              {"author": "user",  "content": {"role": "user",  "parts": [{"text": "First user message"}]}},
              {"author": "agent", "content": {"role": "model", "parts": [{"text": "First agent reply"}]}},
              {"author": "user",  "content": {"role": "user",  "parts": [{"text": "Follow-up user message"}]}}
            ]
          }
        ]
      }
    }
  ]
}
```

## Key Fields

- `eval_cases`: Array of evaluation cases.
- `eval_case_id`: Unique identifier for the evaluation case (optional).
- `prompt`: A single user message — Shape A.
- `agent_data.turns`: Prior conversation turns ending with a user message — Shape B.

## Creating Custom Datasets

You can create custom datasets in two ways:

1. **By Hand**: Copy `basic-dataset.json` as a template and manually add evaluation cases.
2. **Synthesize**: Use the synthetic dataset generation command to generate conversation scenarios:
   ```bash
   agents-cli eval dataset synthesize --count 10
   ```

## Discovering Metrics

You can discover available out-of-the-box evaluation metrics by running:

```bash
agents-cli eval metric list
```

## Beyond Generate and Grade

Once you have a baseline, the eval surface has a few more commands worth knowing about:

- `agents-cli eval compare BASE CAND` — diff two grade-results files (regression check).
- `agents-cli eval analyze RESULTS` — cluster failure modes from a grade-results file.
- `agents-cli eval optimize` — auto-tune your agent's prompts using eval data.

See the [Evaluation Guide](https://google.github.io/agents-cli/guide/evaluation/) for the full surface and metric reference.
