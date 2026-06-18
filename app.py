import streamlit as st

from transcript import get_transcript
from summarizer import summarize

from rag.pipeline import build_rag_pipeline
from rag.retrieval import search_chunks
from rag.qa import answer_question

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    # layout="centered",
    layout="wide",
)

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# UI
# -----------------------------
# st.title("🎥 AI YouTube Video Summarizer")
# st.caption("Paste a YouTube URL to generate an AI-powered summary.")

st.title("🎥 YouTube AI Assistant")
st.caption(
    "Summarize videos and chat with their content using AI."
)


url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

# -----------------------------
# Generate Summary
# -----------------------------
if st.button("🚀 Process Video"):

    if not url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    if "youtube.com" not in url and "youtu.be" not in url:
        st.error("Please enter a valid YouTube URL.")
        st.stop()

    try:
        # Fetch transcript
        with st.spinner("📥 Fetching transcript..."):
            data = get_transcript(url)

        with st.spinner("🧠 Building video knowledge base..."):
            rag_data = build_rag_pipeline(data["text"])

            st.session_state["chunks"] = rag_data["chunks"]
            st.session_state["model"] = rag_data["model"]
            st.session_state["index"] = rag_data["index"]

        st.success("✅ Transcript fetched successfully!")

        # Optional debug preview
        # with st.expander("🔍 Transcript Preview (first 500 characters)"):
        #     st.code(data["text"][:500])

        # Generate summary
        with st.spinner("🤖 Generating AI summary..."):
            summary = summarize(data["text"])

            st.session_state["summary"] = summary
            st.session_state.video_processed = True
            st.session_state.messages = []

        # st.markdown("---")
        # st.subheader("📝 AI Summary")
        # st.markdown(summary)

    except Exception as e:
        st.error(
            "❌ Failed to process this video.\n\n"
            "Possible reasons:\n"
            "- The video has no available transcript.\n"
            "- Captions are disabled.\n"
            "- The transcript is not publicly accessible.\n"
            "- The AI service is temporarily unavailable.\n\n"
            f"Technical details:\n{str(e)}"
        )

if st.session_state.video_processed:

    st.markdown("---")

    st.subheader("📄 AI Summary")

    st.markdown(
        st.session_state["summary"]
    )

if st.session_state.video_processed:

    st.markdown("---")

    st.subheader("💬 Chat With Video")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about this video..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                retrieved_chunks = search_chunks(
                    query=question,
                    model=st.session_state["model"],
                    index=st.session_state["index"],
                    chunks=st.session_state["chunks"],
                    top_k=10
                )

                answer = answer_question(
                    question=question,
                    retrieved_chunks=retrieved_chunks
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )



# -----------------------------
# Footer
# -----------------------------
# st.markdown("---")
# st.caption(
#     "⚠️ Version 1 uses YouTube transcripts. Videos without accessible captions may not be supported."
# )