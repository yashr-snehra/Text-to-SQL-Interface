"""Streamlit frontend. Run: streamlit run app.py  (with api.py served separately)."""
import os

import requests
import streamlit as st

API = os.environ.get("TEXT2SQL_API", "http://localhost:8000")

# Matches make_sample_db.py so the prefilled schema runs against sample.db.
SAMPLE_SCHEMA = """CREATE TABLE artist (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT
);
CREATE TABLE album (
    id INTEGER PRIMARY KEY,
    title TEXT,
    year INTEGER,
    artist_id INTEGER REFERENCES artist(id)
);
CREATE TABLE track (
    id INTEGER PRIMARY KEY,
    title TEXT,
    duration INTEGER,
    genre TEXT,
    album_id INTEGER REFERENCES album(id)
);"""

EXAMPLE_QUESTIONS = [
    "How many tracks are on each album?",
    "List the titles of all tracks by Radiohead.",
    "Which artist has the most albums?",
    "What is the average track duration per genre?",
    "List albums with no tracks.",
]

st.set_page_config(page_title="Text-to-SQL", layout="wide")

# Seed the question once. The widget below is keyed (no value=), so example
# buttons can update it via session_state without Streamlit's "value set both
# ways" warning.
st.session_state.setdefault("question", EXAMPLE_QUESTIONS[0])

with st.sidebar:
    st.header("Text-to-SQL")
    try:
        model = requests.get(f"{API}/health", timeout=5).json().get("model", "?")
        st.caption(f"Model: **{model}** (Ollama)")
    except requests.RequestException:
        st.error("Backend not reachable. Start it with `uvicorn api:app`.")
    mode = st.radio(
        "Mode", ["zero_shot", "few_shot", "few_shot_retry"], index=2,
        help="few_shot_retry runs the repair + self-verify loop.",
    )
    samples = st.slider("Self-consistency samples", 1, 5, 1,
                        help="1 = greedy. >1 samples several queries and votes by result.")
    use_sample_db = st.checkbox("Run against sample.db (has data)", value=True,
                                help="Off: execute against the pasted schema only (no data).")
    st.caption("Try an example:")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

st.title("Natural language -> SQL")

schema_sql = st.text_area("Schema (CREATE TABLE statements)", SAMPLE_SCHEMA, height=240)
question = st.text_input("Question", key="question")

if st.button("Generate SQL", type="primary"):
    # Sending schema_sql executes against the pasted structure (no data). Leave it
    # out to run against sample.db, which has rows.
    payload = {"question": question, "mode": mode, "samples": samples}
    if not use_sample_db:
        payload["schema_sql"] = schema_sql
    try:
        with st.spinner("Asking the model..."):
            r = requests.post(f"{API}/generate", json=payload, timeout=600)
        r.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"API call failed: {exc}")
    else:
        data = r.json()

        if data.get("categories"):
            st.write("Detected: " + "  ".join(f"`{c}`" for c in data["categories"]))

        st.subheader("Generated SQL")
        st.code(data["sql"] or "(empty)", language="sql")

        if data.get("retried"):
            st.info(f"Repaired after a problem. First issue: {data.get('first_error')}")
        if data.get("note"):
            st.caption(data["note"])

        if data.get("executed"):
            cols = data.get("columns") or []
            rows = data.get("rows") or []
            if data.get("structure_only"):
                st.success("Valid against the pasted schema (no data to return).")
            else:
                st.success("Executed against sample.db")
            st.dataframe([dict(zip(cols, row)) for row in rows] if cols else rows)
        elif data.get("error"):
            st.error(f"Execution error: {data['error']}")

        with st.expander("Prompt sent to the model"):
            st.code(data.get("retry_prompt") or data.get("prompt") or "", language="text")
