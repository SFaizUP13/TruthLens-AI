import streamlit as st

from crew.verification_crew import run_claim_analysis
from utils.image_analyzer import analyze_image


st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
)

st.title("🔎 TruthLens AI")

st.write(
    "Multi-Agentic AI system for detecting "
    "misinformation and analyzing media."
)

text_tab, image_tab = st.tabs(
    ["📰 Text Claim Analysis", "🖼️ Image Analysis"]
)


# ==========================================
# TAB 1: EXISTING TEXT CLAIM ANALYSIS
# ==========================================

with text_tab:

    st.subheader("📰 Claim Analysis")

    claim = st.text_area(
        "Enter a news claim:",
        placeholder=(
            "Example: Scientists have discovered "
            "a new planet that can support human life."
        ),
        height=150,
        key="text_claim",
    )

    if st.button("Analyze Claim", key="analyze_claim"):

        if not claim.strip():
            st.warning("Please enter a claim first.")

        else:
            try:
                with st.spinner(
                    "🤖 AI agents are analyzing the claim..."
                ):
                    result = run_claim_analysis(claim)

                st.subheader("🔎 Fact-Check Report")
                st.write(result)

            except Exception as e:
                st.error(
                    f"Claim analysis failed: {str(e)}"
                )


# ==========================================
# TAB 2: NEW IMAGE ANALYSIS
# ==========================================

with image_tab:

    st.subheader("🖼️ Image Analysis")

    st.write(
        "Upload an image to examine its visible content, "
        "readable text, and possible warning signs."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    image_claim = st.text_area(
        "What does the image supposedly show? (Optional)",
        placeholder=(
            "Example: This photograph shows flooding "
            "in Quetta today."
        ),
        height=100,
        key="image_claim",
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button(
            "Analyze Image",
            key="analyze_image",
        ):

            try:
                with st.spinner(
                    "🤖 AI is examining the image..."
                ):
                    result = analyze_image(
                        uploaded_image,
                        image_claim,
                    )

                st.subheader(
                    "🔎 Image Analysis Report"
                )

                st.markdown(result)

                st.info(
                    "This is an AI-generated preliminary "
                    "assessment. It does not prove whether "
                    "the image is genuine or manipulated."
                )

            except Exception as e:
                st.error(
                    f"Image analysis failed: {str(e)}"
                )
