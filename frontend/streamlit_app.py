"""Streamlit UI for the Cloud AI Knowledge Assistant.

A thin client over the FastAPI backend: nothing here talks to AWS, Qdrant,
or the LLM directly - every action goes through the REST API, which keeps
the frontend deployable independently (e.g. Streamlit Community Cloud)
against any backend URL.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

st.set_page_config(
    page_title="Cloud AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

STATUS_BADGES = {
    "uploaded": "🟡 Uploaded",
    "processing": "🔵 Processing",
    "processed": "🟢 Processed",
    "failed": "🔴 Failed",
}


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    return response


def api_post(path: str, **kwargs):
    response = requests.post(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    return response


def api_delete(path: str):
    response = requests.delete(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT)
    return response


def show_error(response: requests.Response) -> None:
    try:
        body = response.json()
        message = body.get("message", response.text)
    except ValueError:
        message = response.text
    st.error(f"❌ {message}")


# --------------------------------------------------------------------------- #
# Sidebar - Upload + Document management
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 Knowledge Assistant")
    st.caption("Cloud-based RAG over your own PDF documents.")

    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None and st.button("Upload", use_container_width=True):
        with st.spinner(f"Uploading and processing {uploaded_file.name}…"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = api_post("/upload", files=files)
                if response.status_code == 201:
                    st.success(f"✅ '{uploaded_file.name}' uploaded and processed.")
                else:
                    show_error(response)
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Could not reach the backend API: {exc}")

    st.divider()
    st.subheader("Your Documents")

    try:
        docs_response = api_get("/documents")
        if docs_response.status_code == 200:
            documents = docs_response.json().get("documents", [])
            if not documents:
                st.info("No documents uploaded yet.")
            for doc in documents:
                badge = STATUS_BADGES.get(doc["status"], doc["status"])
                with st.container(border=True):
                    st.markdown(f"**{doc['document_name']}**")
                    st.caption(f"{badge} · {doc.get('page_count') or '–'} pages · "
                               f"{doc.get('chunk_count') or '–'} chunks")
                    if doc["status"] == "failed" and doc.get("error_message"):
                        st.caption(f"⚠️ {doc['error_message']}")
                    if st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                        del_response = api_delete(f"/documents/{doc['document_id']}")
                        if del_response.status_code == 200:
                            st.success("Deleted.")
                            st.rerun()
                        else:
                            show_error(del_response)
        else:
            show_error(docs_response)
    except requests.exceptions.RequestException as exc:
        st.warning(f"⚠️ Could not reach the backend API at {API_BASE_URL}: {exc}")

# --------------------------------------------------------------------------- #
# Main area - Chat / Ask Questions
# --------------------------------------------------------------------------- #
st.header("💬 Ask a question about your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander(f"📎 Sources ({len(entry['sources'])})"):
                for source in entry["sources"]:
                    st.markdown(
                        f"**{source['document_name']}** — page {source['page_number']} "
                        f"· relevance {source['score']:.2f}"
                    )
                    st.caption(source["text_snippet"])

question = st.chat_input("Ask something about the documents you uploaded…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = api_post("/ask", json={"question": question})
                if response.status_code == 200:
                    payload = response.json()
                    st.write(payload["answer"])
                    sources = payload.get("sources", [])
                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)})"):
                            for source in sources:
                                st.markdown(
                                    f"**{source['document_name']}** — page "
                                    f"{source['page_number']} · relevance {source['score']:.2f}"
                                )
                                st.caption(source["text_snippet"])
                    st.session_state.chat_history.append(
                        {"question": question, "answer": payload["answer"], "sources": sources}
                    )
                else:
                    show_error(response)
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Could not reach the backend API: {exc}")
