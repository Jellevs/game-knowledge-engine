"""Streamlit UI. Talks to the FastAPI service over HTTP.

Deliberately calls the API rather than importing rag directly: the UI stays a
dumb client, the API remains the only thing that knows about retrieval, and the
split mirrors how they would actually be deployed - two containers, not one.
"""


import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PvME RAG", page_icon="⚔")
st.title("RuneScape 3 PvME assistant")
st.caption("Answers come only from the PvME Arch-Glacor guides.")

with st.sidebar:
    st.subheader("Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success('API ONLINE')
        st.write(f"collection: '{health['collection']}'")
        st.write(f"chunks: {health['chunks']}")
        st.write(f"model: '{health['llm_model']}'")
    except Exception  as exc:
        st.error(f"API unreachable: {exc}")

    
question = st.text_input(
    "Question",
    placeholder="Which familiar should I use below 2500% enrage with Necromancy?"

)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        try:
            response = requests.post(
                f"{API_URL}/query", json={"question": question}, timeout=300
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    st.markdown(data["answer"])
    st.caption(f"{data['latency_ms']} ms")

    with st.expander(f"Sources ({len(data['sources'])})"):
        for source in data["sources"]:
            st.write(f"**{source['guide']}** — {source['section']}")