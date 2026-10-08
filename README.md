# PySpark Teacher

Two teachers in one Databricks App. Pick one in the sidebar:

- **🎓 PySpark Teacher**: paste a Spark SQL query, get the idiomatic PySpark DataFrame API
  equivalent, with inline comments mapping each step back to the clause it came from.
- **🔄 Alteryx Migration Teacher**: give it an Alteryx workflow, get functionally equivalent
  Databricks code, a source/target map and the dependencies that will not port.

The work is done by a Databricks-served LLM. Nothing is executed; you only ever get text back.

## Using the PySpark Teacher

1. Paste your query into **Spark SQL** on the left.
2. Optionally add a note (e.g. *use a window function*) to steer the output.
3. Hit **Translate**. The result appears on the right, ready to copy.

## Using the Alteryx Migration Teacher

Built for a like-for-like swap of the Alteryx curation layer: upstream sources and downstream
targets stay as they are, only the logic in between moves to Databricks.

1. Give it the workflow in any of these forms:
   - upload `.yxmd` workflows together with the `.yxmc` macros they call, `.yxwz` apps, or a
     whole `.yxzp` package. Macros you leave out come back as stubs;
   - paste workflow XML (a `.yxmd` is plain XML, so open it in a text editor);
   - paste a single Formula or Filter expression, or describe the workflow in words.
2. Optionally add a note, e.g. *targets live in `main.finance`*.
3. Hit **Translate**. The report on the right has five sections:

| Section | What you get |
|---|---|
| Sources and targets | Every input and output mapped to a Unity Catalog table or Volume path, with its write mode. The columns are the same for every workflow, so the tables stack into one inventory. |
| How the workflow works | A tool-by-tool walkthrough in data-flow order, nested macros included. |
| Databricks code | PySpark, plus SQL where it reads better, with every step tagged with its Alteryx ToolID. |
| Unsupported dependencies | Spatial, geocoding, CASS, predictive, Run Command and similar, each with a Databricks-native alternative and an S/M/L effort estimate. |
| Behaviour differences and parity check | Where Spark semantics differ from Alteryx, and a snippet that compares the new output with the Alteryx one. |

**Download .md** saves the report.

Before anything is sent, the XML loses its canvas layout, every tool's cached output schema and
any stored passwords, and Text Input tables are cut to their first 20 rows. Tool configuration
is sent unchanged. The caption under the upload box shows how much goes to the model.

## Models

Pick one in the sidebar. Each entry knows its endpoint name and the route that serves it:

| Entry | Endpoint | Route |
|---|---|---|
| Qwen 3.5 122B · serving endpoint (default) | `databricks-qwen35-122b-a10b` | `/serving-endpoints` |
| Qwen 3.5 122B · AI Gateway | `system.ai.qwen35-122b-a10b` | `/ai-gateway/mlflow/v1` |

To add a model, open its endpoint under **Serving**, copy `model` and the end of `base_url`
from its **Query** snippet, and add a line to `MODELS` in `translator.py`:

```python
"Llama 3.3 70B": Model("databricks-meta-llama-3-3-70b-instruct", "serving-endpoints"),
```

### Keep max tokens high

Qwen reasons before it answers, and that reasoning comes out of the same token budget. Cut
**Max tokens** too low and the budget is gone before the answer starts: you get
`Model returned an empty response`. If you see that, raise it rather than retrying. Each teacher
has its own default: 16,000 for PySpark and 24,000 for the longer Alteryx report (Qwen 3.5
returns at most 25,000). If an answer gets cut off mid-way, the app tells you.

## Layout

| File | Purpose |
|---|---|
| `teachers.py` | The teachers: system prompts, examples and UI text |
| `translator.py` | Model list, retry loop, and the call to the serving endpoint |
| `alteryx.py` | Reads Alteryx workflows, macros and packages, and strips what the model doesn't need |
| `app.py` | Streamlit UI |
| `app.yaml` | How Databricks starts the process |
| `requirements.txt` | Dependencies installed in the container |

## Running locally

```bash
databricks auth login --host <workspace-url>
pip install -r requirements.txt
streamlit run app.py
```

Instead of the CLI login you can export `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (a personal
access token). The same code path works locally and in the app: `WorkspaceClient()` picks up
your CLI profile or those variables locally, and the service principal's OAuth credentials once
deployed. There is no notebook context in the container, which is why the token is not read from
`dbutils`.

## Deploying

```bash
databricks sync . /Workspace/Users/<you>/pysparkteacher
databricks apps deploy pyspark-teacher --source-code-path /Workspace/Users/<you>/pysparkteacher
```

## Troubleshooting

**`BAD_REQUEST: Cannot create or query foundation model endpoints`**: the workspace itself
blocks these models, not your code. The same error shows up from a plain
`databricks serving-endpoints query`. Deploy to a workspace where foundation models are
enabled.

**`Model returned an empty response`**: almost always too small a **Max tokens**; see above.

**403 on every translation**: the app's service principal needs `CAN_QUERY` on the endpoint
behind the selected model. Add it under the app's resources in the Databricks UI.

**Input too long**: the workflow and its macros don't fit the model's context window (256K
tokens for Qwen 3.5). Send fewer macros at once, or paste only the part of the workflow you
are working on.
