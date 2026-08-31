"""Streamlit front-end for the SQL -> PySpark translator, deployed as a Databricks App."""

from __future__ import annotations

import os

import streamlit as st

from translator import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    TranslationError,
    strip_code_fences,
    stream_translation,
)

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

st.title("🎓 PySpark Teacher")
st.caption("Wklej zapytanie Spark SQL, dostaniesz idiomatyczny odpowiednik w DataFrame API.")

with st.sidebar:
    st.subheader("Ustawienia")
    model = st.text_input("Serving endpoint", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.number_input(
        "Max tokens", min_value=256, max_value=32_000, value=DEFAULT_MAX_TOKENS, step=256
    )
    # Handy when someone reports "it says access denied" - tells you which principal is calling.
    st.divider()
    st.caption(f"Klient: `{os.environ.get('DATABRICKS_CLIENT_ID', 'local profile')}`")

left, right = st.columns(2)

with left:
    query = st.text_area("Spark SQL", value=EXAMPLE_QUERY, height=340)
    note = st.text_input("Dodatkowa uwaga (opcjonalnie)", placeholder="np. użyj window function")
    go = st.button("Przetłumacz", type="primary", use_container_width=True)

with right:
    st.markdown("**PySpark**")
    output = st.empty()

    if go:
        if not query.strip():
            st.warning("Wpisz zapytanie.")
        else:
            buffer = ""
            try:
                with st.spinner("Model pracuje..."):
                    for delta in stream_translation(
                        query,
                        note or None,
                        model=model,
                        max_tokens=int(max_tokens),
                        temperature=temperature,
                    ):
                        buffer += delta
                        output.code(strip_code_fences(buffer), language="python")
                st.session_state["last_code"] = strip_code_fences(buffer)
            except TranslationError as exc:
                st.error(f"Tłumaczenie się nie udało: {exc}")
            except Exception as exc:  # surface endpoint/permission errors instead of a blank page
                st.error(f"{type(exc).__name__}: {exc}")
    elif "last_code" in st.session_state:
        output.code(st.session_state["last_code"], language="python")
