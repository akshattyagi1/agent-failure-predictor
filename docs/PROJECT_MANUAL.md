# Agent Failure Predictor: project manual

This is a guide to what the project does, what happens during a run, how its ML component works, and how to publish it responsibly on GitHub.

## 1. The project in one sentence

We run a local tool-using LLM agent, record what happens during its work, and use those runtime records to predict whether the agent is likely to fail before the run has finished.

There are two separate systems:

1. **The agent** tries to answer a task by calling a local Ollama model and local tools.
2. **The predictor** reads telemetry from the agent and estimates failure risk.

The predictor is not the LLM. It is a small machine-learning model that we trained ourselves using this project's telemetry.

```text
User task
  -> local Ollama agent decides what to do
  -> agent calls a local tool when needed
  -> every event is written as telemetry
  -> feature tracker turns the events into numbers
  -> failure predictor returns a risk probability
  -> agent still continues normally (shadow mode)
```

## 2. What happens when you run the agent

Example command:

```powershell
agent-run "What is 18 multiplied by 7?" --model llama3.2:3b
```

The entry point is `agent_failure_predictor/runner.py`.

1. The runner creates a unique `run_id`. A run means one attempt to complete one user task.
2. It chooses a local environment, which contains deterministic demo documents and a small read-only SQLite database.
3. It sends the task, system instructions, and tool definitions to Ollama.
4. The LLM either returns an answer or requests a tool.
5. The runner performs only allow-listed local tools:
   - `calculate`
   - `search_documents`
   - `describe_database_schema`
   - `query_database` (read-only `SELECT` only)
6. The tool result is given back to the LLM, which can decide its next action.
7. The loop stops when the LLM answers, the agent hits its step limit, or an unrecoverable error occurs.
8. A final run summary is written.

For database questions, the agent must call `describe_database_schema` before running SQL. This prevents it from guessing columns such as `customer_name` when the database actually uses `holder_name` or `customer`.

## 3. What telemetry is

Telemetry means structured records about how the agent behaved. It is not a hidden internal model state and it is not the final judgement of whether an answer was correct.

Two JSONL files are written locally:

```text
telemetry/steps.jsonl  one record for each LLM call or tool call
telemetry/runs.jsonl   one summary record when a run finishes
```

JSONL means “JSON Lines”: every line is one valid JSON object. It is useful because new events can be appended without rewriting older records.

### Step events

A step event can record information such as:

- `run_id`: which run this belongs to
- `event_type`: `llm_call` or `tool_call`
- latency and token counts
- the tool name and whether it succeeded
- retry count and error category
- safe diagnostic metadata about tool arguments and errors
- optional live prediction probability and label

### Run events

A run event records the end-of-run summary, such as total latency, number of LLM calls, number of tool errors, whether a final answer was generated, and whether the agent execution itself failed.

Raw telemetry is ignored by Git because it is generated data and can become large quickly.

## 4. How runs become an ML dataset

The project has two sources of labelled data.

### Synthetic data

`failure_injection.py` deliberately injects controlled problems such as latency, tool failures, retries, empty results, or degraded results. This lets us create many reproducible failure trajectories while building the pipeline.

Synthetic labels answer: “Did the injected/operational run eventually fail?”

### Real local-agent data

`evaluate_real_runs.py` sends version-controlled tasks to the actual local Ollama agent. Each task has a deterministic evaluation rule, for example:

```json
{"type": "numeric", "expected": 126}
```

or:

```json
{"type": "contains_all", "expected": ["lina", "3200"]}
```

The evaluator compares the final answer with that rule. This provides an answer-quality label even if the agent itself technically completed without a Python exception.

The generated experiment contains:

```text
raw/steps.jsonl          raw step telemetry
raw/runs.jsonl           raw run summaries
raw/evaluations.jsonl    deterministic task judgements
processed/checkpoints_engineered.csv
processed/runs.csv
real_evaluation/         report, metrics, audit table, chart
```

## 5. Feature engineering and leakage safety

Machine-learning models need numeric inputs. `features/engineering.py` converts each checkpoint into features such as:

- `checkpoint_index`
- cumulative latency
- LLM call and tool call counts
- retries and tool errors so far
- input/output tokens so far
- consecutive or recent tool errors
- repeated or unique tool use
- empty/malformed tool results

The important rule is **no future information**. At checkpoint 2, the feature row can use only events already seen by checkpoint 2. It cannot use the final answer, a later error, or the eventual success/failure label as an input.

The eventual label is attached only as the target the model should learn to predict.

## 6. How the predictor is trained

The current served predictor is a Logistic Regression model. Logistic Regression is a good first model because it is fast, inspectable, and provides probabilities.

Training does this:

```text
checkpoint features + eventual outcome label
  -> training/validation split
  -> fit Logistic Regression
  -> select an alert threshold on validation data
  -> evaluate on data that was never used for fitting
```

The model returns a probability such as `0.752`. The threshold turns that probability into either `high_risk` or `low_risk`.

Threshold choice matters:

- Lower threshold: catches more failures, but produces more false alerts.
- Higher threshold: produces fewer alerts, but misses more failures.

## 7. Why task-template splitting matters

An LLM can behave differently each time, so we repeat task templates. For example, three runs may use the same prompt asking for an account balance.

Those repeats are similar. If one repeat is used for training and another for validation, the score can look unrealistically good because the model has already seen a nearly identical workflow.

The comparison pipeline therefore splits by `task_template_id`: every repeat of one template is entirely in training, validation, or test. It never crosses partitions.

## 8. Live prediction and the FastAPI service

The API is in `api/main.py`.

```powershell
uvicorn api.main:app --reload
```

It exposes:

- `GET /health`: confirms that the model bundle loaded.
- `POST /predict`: accepts the current telemetry feature values and returns a probability, risk label, threshold, and top contributing features.

When the agent is run with `--prediction-endpoint http://127.0.0.1:8000`, every new telemetry event is sent to the API.

This is **shadow mode**. The predictor reports risk but does not stop, modify, or control the agent. That is intentional: first we measure the predictor honestly; only later would we consider interventions such as retries, handoffs, or human review.

## 9. Explanations

The project uses SHAP offline for the Logistic Regression model and computes compatible linear contributions at API runtime.

An explanation does not mean “this feature caused the failure.” It means “this feature pushed the model's prediction up or down relative to its baseline.” For example, many retries and high cumulative latency can increase the model's estimated risk.

## 10. Reading the final held-out evaluation

The final V3 experiment ran 300 times across 30 new task templates, each repeated 10 times. These templates were not used to train the current model or choose its threshold.

| Metric | Result |
| --- | ---: |
| Precision | 59.3% |
| Recall | 65.0% |
| Specificity | 76.6% |
| Accuracy | 72.7% |
| False-alert rate | 23.4% |

Interpretation:

- When the model alerted, it was correct about 59% of the time.
- It caught 65% of genuinely failing runs.
- It did not alert on 76.6% of successful runs.
- It still falsely alerted on 23.4% of successful runs.

This is a credible research result, but not a deployable universal failure predictor. V3 must remain evaluation-only. If it is used for training later, we need a new V4 held-out task bank.

## 11. Using another local Ollama model

You can use any locally pulled Ollama model:

```powershell
ollama list
agent-run "List customers with a balance over 1000." --model <model-name>
```

The agent will work best when the model can call tools reliably. The predictor can technically score another model's telemetry because the feature schema stays the same, but the probabilities are not automatically calibrated for a new LLM. To make reliable claims about that model, collect and label its runs, train/validate on development data, and use a fresh held-out task bank.

## 12. How to publish this repository to GitHub

The project is already staged with the intended content. It includes code, tests, docs, versioned task banks, and the tiny runtime model bundle. It excludes generated telemetry, SQLite databases, 300-run data, and bulky artifacts through `.gitignore`.

### Step A: configure your Git identity once

Use your own name and the email connected to GitHub:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Check it:

```powershell
git config --global --get user.name
git config --global --get user.email
```

### Step B: create an empty GitHub repository

On GitHub:

1. Click **New repository**.
2. Use a name such as `agent-failure-predictor`.
3. Choose **Public** for a portfolio project.
4. Do **not** initialize it with a README, `.gitignore`, or license, because this local project already has those files.
5. Create the repository and copy its HTTPS URL.

### Step C: create the first local commit

From the project folder:

```powershell
git commit -m "Initial release: telemetry-driven agent failure predictor"
git branch -M main
```

### Step D: connect and push

Replace the URL below with your copied GitHub URL:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/agent-failure-predictor.git
git push -u origin main
```

GitHub may open a browser to authenticate. Complete the login as your GitHub account.

### Step E: later updates

```powershell
git status
git add <specific-files>
git commit -m "Describe the change"
git push
```

Avoid `git add -f` for ignored experiment logs. Ignored data is excluded for a reason.

## 13. Make the GitHub page look strong

After pushing, improve the GitHub repository page:

1. Set the repository description to: `Telemetry-first failure-risk prediction for local tool-using LLM agents.`
2. Add topics: `machine-learning`, `llm-agents`, `ollama`, `fastapi`, `observability`, `scikit-learn`, `shap`, `agent-evaluation`.
3. Pin the repository on your GitHub profile.
4. Keep the root README as the landing page; it already contains the architecture, quick start, results, and limitations.
5. Add two or three screenshots later: FastAPI `/docs`, a shadow alert in the terminal, and the held-out evaluation report. Do not upload raw telemetry or private prompts.
6. Add a license before inviting reuse. MIT is a common permissive choice; choose it only if you want others to reuse the code under those terms.
7. Keep commits focused and readable. Good examples are `Add held-out evaluation task bank` and `Document leakage-safe template split`.

## 14. Glossary

- **Agent:** the program using an LLM and tools to perform a task.
- **Run:** one complete attempt by the agent to perform one task.
- **Checkpoint:** the state after one observed LLM/tool event where a risk prediction can be made.
- **Telemetry:** structured operational records from a run.
- **Feature:** one numeric input used by the ML model.
- **Label:** the known outcome the model learns to predict, such as eventual failure.
- **False positive / false alert:** the predictor alerts but the run succeeds.
- **False negative / missed failure:** the predictor does not alert but the run fails.
- **Calibration:** whether predicted probabilities match observed frequencies over many runs.
- **Data leakage:** accidentally giving training access to information it would not have at prediction time, or to near-duplicate test examples.
- **Held-out evaluation:** final testing on data never used to make modelling decisions.
