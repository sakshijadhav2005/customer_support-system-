from dotenv import load_dotenv

load_dotenv()  # must run before the graph is imported

import streamlit as st  # noqa: E402

import ingest  # noqa: E402
import kb  # noqa: E402

st.set_page_config(page_title="Customer Support", layout="wide")

SAMPLES = [
    "What is your return policy?",
    "How do I reset my password?",
    "Where is my order #A1023?",
    "My order is 3 weeks late and I want a refund!",
    "Do you sell gift cards in Antarctica?",
]
ROUTE_LABELS = {
    "escalation": "Escalation",
    "off_topic": "Off-topic (polite redirect, no model call)",
    "retrieval": "FAQ retrieval, then response generation",
    "direct": "Direct response generation",
}
URGENCY_COLORS = {"low": "green", "medium": "orange", "high": "red"}


@st.cache_resource
def load_graph():
    from graph import app  # built once, reused for every request

    return app


graph = load_graph()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("latest", None)
st.session_state.setdefault("upload_status", None)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("pending", None)


def run_query(query: str) -> None:
    """Save the question first so it is never lost, then run the graph."""
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        result = graph.invoke(
            {
                "user_query": query,
                "category": "",
                "urgency": "",
                "retrieved_docs": [],
                "response": "",
                "route": "",
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant",
             "content": f"Sorry, something went wrong ({type(exc).__name__}). Please try again."}
        )
        return
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    st.session_state.latest = result


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📚 Knowledge base")
    st.caption("Add PDF, TXT or DOCX files (max 5 MB). The bot will answer from them.")
    uploaded = st.file_uploader(
        "Upload a document",
        type=None,  # validated in ingest so the user sees our own messages
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded is not None and st.button(
        "Add to knowledge base", type="primary", use_container_width=True
    ):
        try:
            count = ingest.process_upload(uploaded.name, uploaded.getvalue())
            st.session_state.upload_status = (
                "success", f"Upload complete: {count} sections added to the knowledge base")
        except ingest.UploadError as exc:
            st.session_state.upload_status = ("error", str(exc))
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.subheader("Uploaded documents")
    docs = kb.list_documents()
    if not docs:
        st.caption("No documents uploaded yet.")
    for doc in docs:
        with st.container(border=True):
            info_col, delete_col = st.columns([5, 1], vertical_alignment="center")
            info_col.markdown(f"**📄 {doc['name']}**")
            if delete_col.button("🗑️", key=f"rm_{doc['name']}", help="Delete this document"):
                kb.remove_document(doc["name"])
                st.session_state.upload_status = (
                    "success", f"Deleted {doc['name']} and removed its sections from the knowledge base")
                st.rerun()
            st.caption(
                f"{doc['type']} · {doc['size'] / 1024:.1f} KB · "
                f"{doc['chunk_count']} sections · {doc['added']}"
            )

    st.divider()
    st.subheader("Sample queries")
    for i, sample in enumerate(SAMPLES):
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending = sample

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest = None
        st.rerun()

# ---------------- Main area ----------------
st.title("Customer Support")
chat_col, details_col = st.columns([2, 1])

typed = st.chat_input("Type your question")
query = st.session_state.pending or typed
st.session_state.pending = None

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if query and query.strip():
        with st.chat_message("user"):  # shown right away, before the reply is ready
            st.write(query.strip())
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                run_query(query.strip())
            st.write(st.session_state.messages[-1]["content"])

with details_col:
    st.subheader("Behind the scenes")
    status = st.session_state.upload_status
    if status:
        (st.success if status[0] == "success" else st.error)(status[1])
    latest = st.session_state.latest
    if latest is None:
        st.caption("Send a message to see how it was handled.")
    else:
        st.markdown(f"**Category:** {latest['category'].replace('_', ' ').title()}")
        color = URGENCY_COLORS.get(latest["urgency"], "gray")
        st.markdown(f"**Urgency:** :{color}[{latest['urgency']}]")
        st.markdown(f"**Route:** {ROUTE_LABELS.get(latest['route'], latest['route'])}")
        st.markdown("**Retrieved entries:**")
        if latest["retrieved_docs"]:
            for d in latest["retrieved_docs"]:
                question, _, answer = d.partition("\nA: ")
                with st.expander(question.removeprefix("Q: "), expanded=True):
                    st.write(answer)
        else:
            st.caption("No FAQ entry matched.")
