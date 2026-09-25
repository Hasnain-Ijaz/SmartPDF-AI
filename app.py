import streamlit as st
from ai_workflow import run_study_workflow, render_final_pack

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Pack Generator")
st.caption("Personalized study material generated through a multi-stage AI workflow.")

with st.sidebar:
    st.header("Study Settings")
    subject = st.text_input("Subject", placeholder="e.g. Artificial Intelligence")
    topic = st.text_input("Topic", placeholder="e.g. Generative AI")
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    study_time = st.selectbox(
        "Study Time",
        ["30 minutes", "1 hour", "2 hours", "3 hours", "4+ hours"],
    )
    language = st.selectbox("Language", ["English", "Roman Urdu", "Urdu"])
    goal = st.text_area(
        "Learning Goal",
        placeholder="What do you want to achieve?",
        height=100,
    )

    generate = st.button("🚀 Generate Study Pack", use_container_width=True)

if generate:
    if not subject.strip() or not topic.strip():
        st.error("Please enter both Subject and Topic.")
    else:
        try:
            with st.spinner("Running the AI study workflow..."):
                result = run_study_workflow(
                    subject=subject.strip(),
                    topic=topic.strip(),
                    level=level,
                    study_time=study_time,
                    language=language,
                    goal=goal.strip(),
                )

            st.session_state["study_pack"] = result
            st.success("Study pack generated successfully!")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

result = st.session_state.get("study_pack")

if result:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 Final Study Pack", "⚙️ Workflow Stages", "🔍 Review", "🧾 Raw JSON"]
    )

    with tab1:
        markdown = render_final_pack(result["final"])
        st.markdown(markdown)

        st.download_button(
            "⬇️ Download Study Pack",
            data=markdown,
            file_name="ai_study_pack.md",
            mime="text/markdown",
        )

    with tab2:
        stages = [
            ("1. Planning", result.get("planning")),
            ("2. Content Generation", result.get("content")),
            ("3. Assessment", result.get("assessment")),
            ("4. Review", result.get("review")),
            ("5. Refinement", result.get("final")),
        ]

        for title, data in stages:
            with st.expander(title):
                st.json(data)

    with tab3:
        st.json(result.get("review", {}))

    with tab4:
        st.json(result)
else:
    st.info("Enter your study details from the sidebar and click Generate Study Pack.")
