import streamlit as st

from crew.verification_crew import run_claim_analysis


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="centered"
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("🔎 TruthLens AI")

st.write(
    "Multi-Agentic AI system for detecting "
    "misinformation and manipulated media."
)


# --------------------------------------------------
# Claim input
# --------------------------------------------------

st.subheader("📰 Claim Analysis")

claim = st.text_area(

    "Enter a news claim:",

    placeholder=(
        "Example: Scientists have discovered "
        "a new planet that can support human life."
    ),

    height=150
)


# --------------------------------------------------
# Analyze claim
# --------------------------------------------------

if st.button("Analyze Claim"):

    if not claim.strip():

        st.warning(
            "Please enter a claim first."
        )

        st.stop()

    try:

        with st.spinner(
            "🤖 Claim Analyst is analyzing..."
        ):

            result = run_claim_analysis(
                claim
            )

        st.subheader(
            "🔎 Claim Analysis"
        )

        st.write(result)

    except Exception as e:

        st.error(
            f"Something went wrong: {str(e)}"
        )
