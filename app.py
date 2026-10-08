"""Streamlit front-end for the teachers in `teachers.py`, deployed as a Databricks App."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import alteryx
from teachers import TEACHERS
from translator import MODELS, translate

README = Path(__file__).parent / "README.md"

st.set_page_config(page_title="PySpark Teacher", page_icon="🎓", layout="wide")


@st.dialog("README", width="large")
def show_readme() -> None:
    """Render README.md in a modal so the docs travel with the deployed app."""
    try:
        st.markdown(README.read_text(encoding="utf-8"))
    except OSError as exc:
        st.error(f"Could not read README.md: {exc}")


with st.sidebar:
    st.subheader("Settings")
    teacher_key = st.selectbox(
        "Teacher", list(TEACHERS), format_func=lambda key: TEACHERS[key].title
    )
    teacher = TEACHERS[teacher_key]
    model = MODELS[st.selectbox("Model", list(MODELS))]
    st.caption(f"`{model.name}` via `/{model.route}`")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.number_input(
        "Max tokens",
        min_value=256,
        max_value=32_000,
        value=teacher.max_tokens,
        step=256,
        key=f"max_tokens_{teacher_key}",
    )

header, readme_button = st.columns([5, 1])
with header:
    st.title(teacher.title)
    st.caption(teacher.caption)
with readme_button:
    if st.button("📖 README", use_container_width=True):
        show_readme()

left, right = st.columns(2)

with left:
    text = st.text_area(
        teacher.input_label, value=teacher.example, height=340, key=f"input_{teacher_key}"
    )
    source = text
    if teacher_key == "alteryx":
        files = st.file_uploader(
            "...or upload workflows, macros or a .yxzp package",
            type=alteryx.UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        try:
            if files:
                source = alteryx.read_uploads((f.name, f.getvalue()) for f in files)
            else:
                source = alteryx.compact(text)
        except ValueError as exc:
            st.error(str(exc))
            source = ""
        if source and source != text:
            origin = f"{len(files)} uploaded file(s), " if files else ""
            st.caption(
                f"Sending {origin}{len(source):,} characters (≈{len(source) // 4:,} tokens) "
                "without canvas layout, cached schemas or passwords."
            )
    note = st.text_input(
        "Extra note (optional)", placeholder=teacher.note_placeholder, key=f"note_{teacher_key}"
    )
    go = st.button("Translate", type="primary", use_container_width=True)

answer_key = f"answer_{teacher_key}"

with right:
    st.markdown(f"**{teacher.output_label}**")

    if go and source.strip():
        try:
            with st.spinner("Working..."):
                st.session_state[answer_key] = translate(
                    source,
                    teacher,
                    note or None,
                    model=model,
                    max_tokens=int(max_tokens),
                    temperature=temperature,
                )
        except Exception as exc:  # show the real error instead of a blank page
            st.session_state.pop(answer_key, None)
            st.error(f"{type(exc).__name__}: {exc}")
    elif go:
        st.warning("Enter something to translate first.")

    if answer := st.session_state.get(answer_key):
        if answer.truncated:
            st.warning(
                "The answer hit **Max tokens** and stops mid-way. "
                "Raise it in the sidebar and run again."
            )
        if teacher.output == "python":
            st.code(answer.text, language="python")
        else:
            st.markdown(answer.text)
            st.download_button(
                "⬇️ Download .md",
                answer.text,
                file_name="migration-report.md",
                mime="text/markdown",
            )
