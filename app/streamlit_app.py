import os
import sys

# -- Resolve project root and add source paths ---------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "retrieval"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "generation"))
# ------------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import streamlit as st
from retriever import load_retriever
from generator import ask


# -- Page configuration --------------------------------------------------------
st.set_page_config(
    page_title="Luminary Knowledge Assistant",
    page_icon="L",
    layout="centered"
)

# -- Pin the last sidebar elements to the bottom -------------------------------
st.markdown(
    """
    <style>
    /* Force the full flex chain down to the vertical block */
    [data-testid="stSidebarUserContent"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {
        height: 100%;
        flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -- Load the retriever once and cache it --------------------------------------
@st.cache_resource
def get_retriever(source_filter):
    """
    Load the retriever once and keep it in memory across reruns.
    Streamlit reruns the whole script on every interaction, so
    caching prevents reloading the embedding model every time.
    """
    return load_retriever(source_filter=source_filter)


# -- Header --------------------------------------------------------------------
st.title("Luminary Knowledge Assistant")
st.caption(
    "Ask questions about Luminary Data & AI's client engagements, "
    "solution proposals, and the UK Government Digital Service blog."
)


# -- Sidebar controls ----------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    source_choice = st.radio(
        "Knowledge source",
        options=["All sources", "Luminary documents only", "GDS blog only"],
        index=0,
        help="Filter which part of the knowledge base to search."
    )

    filter_map = {
        "All sources": None,
        "Luminary documents only": "luminary",
        "GDS blog only": "gds"
    }
    source_filter = filter_map[source_choice]

    # Push the buttons toward the bottom of the sidebar
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)

    # Toggle state for the About panel
    if "show_about" not in st.session_state:
        st.session_state.show_about = False

    # The About content appears ABOVE the buttons when toggled on
    if st.session_state.show_about:
        st.info(
            "This is a Retrieval-Augmented Generation (RAG) assistant. "
            "It retrieves relevant chunks from a knowledge base of 110 "
            "documents and uses Claude to generate grounded, cited answers."
        )

# Two buttons side by side
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("About"):
            st.session_state.show_about = not st.session_state.show_about
            st.rerun()

    with col2:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# -- Initialise conversation history -------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -- Display existing conversation ---------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this was an assistant message with sources, show them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")


# -- Handle new user input -----------------------------------------------------
user_question = st.chat_input("Ask a question...")

if user_question:
    # Add user message to history and display it
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and display the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            retriever = get_retriever(source_filter)
            result = ask(user_question, retriever)

            answer = result["answer"]
            # Deduplicate sources while preserving order
            sources = list(dict.fromkeys(result["sources"]))

        st.markdown(answer)

        with st.expander("Sources used"):
            for source in sources:
                st.markdown(f"- `{source}`")

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })