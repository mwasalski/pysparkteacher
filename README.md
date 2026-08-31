# PySpark Teacher

Paste a Spark SQL query, get the idiomatic PySpark DataFrame API equivalent — with inline
comments mapping each step back to the clause it came from.

Runs as a Databricks App. The translation is done by a Databricks-served LLM; nothing is
executed, you only ever get source code back.

## Using it

1. Paste your query into **Spark SQL** on the left.
2. Optionally add a note (e.g. *use a window function*) to steer the output.
3. Hit **Translate**. The result appears on the right, ready to copy.

Model, temperature and token budget live in the sidebar.

### Keep max tokens high

The default model reasons before it answers, and that reasoning comes out of the same token
budget. Cut **Max tokens** too low and the budget is gone before the answer starts — you get
`Model returned an empty response`. If you see that, raise it rather than retrying.

## Layout

| File | Purpose |
|---|---|
| `translator.py` | Prompt, retry loop, and the call to the serving endpoint |
| `app.py` | Streamlit UI |
| `app.yaml` | How Databricks starts the process |
| `requirements.txt` | Dependencies installed in the container |

## Running locally

```bash
databricks auth login --host <workspace-url>
pip install -r requirements.txt
streamlit run app.py
```

The same code path works locally and in the app: `WorkspaceClient()` picks up your CLI
profile locally and the service principal's OAuth credentials once deployed. There is no
notebook context in the container, which is why the token is not read from `dbutils`.

## Deploying

```bash
databricks sync . /Workspace/Users/<you>/pysparkteacher
databricks apps deploy pyspark-teacher --source-code-path /Workspace/Users/<you>/pysparkteacher
```

## Configuration

| Variable | Default |
|---|---|
| `SERVING_MODEL` | `system.ai.qwen35-122b-a10b` |

## Troubleshooting

**`BAD_REQUEST: Cannot create or query foundation model endpoints`** — the workspace itself
blocks these models, not your code. The same error shows up from a plain
`databricks serving-endpoints query`. Deploy to a workspace where foundation models are
enabled.

**`Model returned an empty response`** — almost always too small a **Max tokens**; see above.

**403 on every translation** — the app's service principal needs `CAN_QUERY` on the endpoint.
Add it under the app's resources in the Databricks UI.
