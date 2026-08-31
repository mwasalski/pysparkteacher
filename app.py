"""Streamlit front-end for the SQL -> PySpark translator, deployed as a Databricks App."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from translator import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, translate_sql_to_pyspark

README = Path(__file__).parent / "README.md"

EXAMPLE_QUERY = """SELECT
    c.country,
    COUNT(*) AS orders,
    SUM(o.amount) AS revenue
FROM main.sales.orders o
JOIN main.sales.customers c ON c.id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.country
HAVING SUM(o.amount) > 10000
ORDER BY revenue DESC
LIMIT 20"""

st.set_page_config(page_title="PySpark Teacher", page_icon="🎓", layout="wide")


@st.dialog("README", width="large")
def show_readme() -> None:
    """Render README.md in a modal so the docs travel with the deployed app."""
    try:
        st.markdown(README.read_text(encoding="utf-8"))
    except OSError as exc:
        st.error(f"Could not read README.md: {exc}")


header, readme_button = st.columns([5, 1])
with header:
    st.title("🎓 PySpark Teacher")
    st.caption("Paste a Spark SQL query, get the idiomatic DataFrame API equivalent.")
with readme_button:
    if st.button("📖 README", use_container_width=True):
        show_readme()

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.number_input(
        "Max tokens", min_value=256, max_value=32_000, value=DEFAULT_MAX_TOKENS, step=256
    )

left, right = st.columns(2)

with left:
    query = st.text_area("Spark SQL", value=EXAMPLE_QUERY, height=340)
    note = st.text_input("Extra note (optional)", placeholder="e.g. use a window function")
    go = st.button("Translate", type="primary", use_container_width=True)

with right:
    st.markdown("**PySpark**")

    if go and query.strip():
        try:
            with st.spinner("Working..."):
                st.session_state["code"] = translate_sql_to_pyspark(
                    query,
                    note or None,
                    model=model,
                    max_tokens=int(max_tokens),
                    temperature=temperature,
                )
        except Exception as exc:  # show the real error instead of a blank page
            st.session_state.pop("code", None)
            st.error(f"{type(exc).__name__}: {exc}")
    elif go:
        st.warning("Enter a query first.")

    if "code" in st.session_state:
        st.code(st.session_state["code"], language="python")
