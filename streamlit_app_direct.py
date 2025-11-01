"""Streamlit UI for Samarth Direct Interpretation Q&A System."""

import streamlit as st
import sys
sys.path.append('.')

from src.app_direct import SamarthDirectApp

# Page configuration
st.set_page_config(
    page_title="Samarth - Direct Interpretation",
    page_icon="🌾",
    layout="wide"
)

# Initialize app
@st.cache_resource
def get_app():
    return SamarthDirectApp()

app = get_app()

# Title
st.title("🌾 Samarth Q&A System")
st.subheader("Direct LLM Interpretation Approach")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Direct Interpretation Approach:**

    1. 🔍 Discover datasets from data.gov.in
    2. 📥 Fetch RAW data (no transformation)
    3. 🤖 Send raw data directly to LLM
    4. 💡 LLM interprets and answers

    **Benefits:**
    - ✅ 100% dataset loading (no transformation failures)
    - ✅ LLM handles messy data intelligently
    - ✅ More flexible than rigid schemas

    **Trusted Data Sources:**
    - India Meteorological Department (IMD)
    - Ministry of Agriculture & Farmers Welfare
    """)

    st.divider()

    # Catalog stats
    st.header("📊 Dataset Catalog")
    stats = app.get_catalog_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Climate", stats['climate_datasets'])
    with col2:
        st.metric("Agriculture", stats['agriculture_datasets'])
    st.metric("Total Datasets", stats['total_datasets'])

    st.divider()

    # Settings
    st.header("⚙️ Settings")
    max_datasets = st.slider(
        "Max datasets to use",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of datasets to send to LLM"
    )
    max_rows = st.slider(
        "Max rows per dataset",
        min_value=100,
        max_value=2000,
        value=500,
        help="Maximum rows to send per dataset"
    )
    auto_discover = st.checkbox(
        "Auto-discover datasets",
        value=True,
        help="Automatically find and add new datasets based on your question"
    )

# Initialize session state
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None
if 'run_query' not in st.session_state:
    st.session_state.run_query = False

# Example questions
st.markdown("### 💡 Example Questions")
st.markdown("*Click to run directly, or type your own question below*")
examples = [
    "What was the average annual rainfall in Odisha in 1951?",
    "Compare rainfall patterns between Odisha and Punjab in the 1950s",
    "What are the rainfall patterns in Andaman and Nicobar Islands from 1951 to 1960?",
    "How does Maharashtra's rainfall compare to Punjab?"
]

# Create columns for example buttons
cols = st.columns(2)
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.selected_example = example
            st.session_state.run_query = True

# Main question input
question = st.text_area(
    "Ask a question about Indian agriculture and climate:",
    value=st.session_state.selected_example if st.session_state.selected_example else "",
    key="question_text_area",
    height=100,
    placeholder="E.g., Compare rainfall and crop production trends in Maharashtra and Punjab..."
)

# Submit button OR auto-run from example
submit_clicked = st.button("🔍 Get Answer", type="primary", use_container_width=True)

# Check if we should run the query
should_run = submit_clicked or st.session_state.run_query

if should_run:
    # Use the question from text area, or from selected example
    query_text = question if question else st.session_state.selected_example

    if not query_text:
        st.warning("Please enter a question.")
    else:
        # Reset the run_query flag
        st.session_state.run_query = False
        st.session_state.selected_example = query_text  # Keep the question visible

        # Show progress
        with st.spinner("🔍 Discovering datasets and analyzing..."):
            try:
                # Get answer
                result = app.answer_question(
                    question=query_text,
                    auto_discover=auto_discover,
                    max_datasets=max_datasets,
                    max_rows_per_dataset=max_rows
                )

                # Show discovery notification
                if result.get('discovered_new'):
                    st.success("✅ Found and added new datasets from data.gov.in!")

                # Display answer
                st.markdown("### 📝 Answer")
                st.markdown(result['answer'])

                # Display datasets used
                if result['datasets_used']:
                    st.markdown("### 📚 Datasets Used")
                    for ds in result['datasets_used']:
                        st.markdown(f"- {ds}")

                # Display sources
                if result['sources']:
                    with st.expander("🔗 Data Sources"):
                        for source_id in result['sources']:
                            st.code(source_id)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by Gemini Flash Lite | Data from data.gov.in</small>
</div>
""", unsafe_allow_html=True)
