import hashlib

import streamlit as st

from crew.verification_crew import run_claim_analysis
from utils.image_analyzer import (
    analyze_image,
    extract_main_claim,
)


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
    [
        "📰 Text Claim Analysis",
        "🖼️ Image Analysis",
    ]
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

    if st.button(
        "Analyze Claim",
        key="analyze_claim",
    ):

        if not claim.strip():

            st.warning("Please enter a claim first.")

        else:

            try:

                with st.spinner(
                    "🤖 AI agents are analyzing the claim..."
                ):

                    result = run_claim_analysis(claim)

                st.subheader("🔎 Fact-Check Report")
                st.markdown(result)

            except Exception as e:

                st.error(
                    f"Claim analysis failed: {str(e)}"
                )


# ==========================================
# TAB 2: IMAGE ANALYSIS AND FACT-CHECKING
# ==========================================

with image_tab:

    st.subheader("🖼️ Image Analysis")

    st.write(
        "Upload an image to extract its text, identify "
        "a factual claim, and send that claim to the "
        "TruthLens AI fact-checking agents."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    # Reset previous results when the uploaded image changes.
    if uploaded_image is not None:

        image_fingerprint = hashlib.sha256(
            uploaded_image.getvalue()
        ).hexdigest()

        previous_fingerprint = st.session_state.get(
            "analyzed_image_fingerprint"
        )

        if image_fingerprint != previous_fingerprint:

            st.session_state.pop(
                "image_analysis_report",
                None,
            )

            st.session_state.pop(
                "image_fact_check_report",
                None,
            )

            st.session_state.pop(
                "image_claim_for_fact_check",
                None,
            )

            st.session_state[
                "analyzed_image_fingerprint"
            ] = image_fingerprint

    else:

        st.session_state.pop(
            "analyzed_image_fingerprint",
            None,
        )

        st.session_state.pop(
            "image_analysis_report",
            None,
        )

        st.session_state.pop(
            "image_fact_check_report",
            None,
        )

        st.session_state.pop(
            "image_claim_for_fact_check",
            None,
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
                    "🤖 AI is analyzing the image..."
                ):

                    image_report = analyze_image(
                        uploaded_image
                    )

                st.session_state[
                    "image_analysis_report"
                ] = image_report

                extracted_claim = extract_main_claim(
                    image_report
                )

                st.session_state[
                    "image_claim_for_fact_check"
                ] = extracted_claim

                # A new image analysis invalidates the
                # previous claim-verification report.
                st.session_state.pop(
                    "image_fact_check_report",
                    None,
                )

            except Exception as e:

                st.error(
                    f"Image analysis failed: {str(e)}"
                )

        # Display the image-analysis report.
        if "image_analysis_report" in st.session_state:

            st.divider()

            st.subheader(
                "🔎 Image Analysis Report"
            )

            st.markdown(
                st.session_state[
                    "image_analysis_report"
                ]
            )

            st.info(
                "Image analysis identifies visible content "
                "and possible warning signs. It does not "
                "establish whether an image is authentic."
            )

            st.divider()

            st.subheader(
                "📝 Review the Extracted Claim"
            )

            st.write(
                "Check the extracted claim below. "
                "You can correct it before fact-checking."
            )

            claim_for_fact_check = st.text_area(
                "Claim to verify:",
                key="image_claim_for_fact_check",
                height=120,
                placeholder=(
                    "The factual claim extracted from the image "
                    "will appear here."
                ),
            )

            if st.button(
                "🔎 Fact-check This Claim",
                key="fact_check_image_claim",
            ):

                if not claim_for_fact_check.strip():

                    st.warning(
                        "No claim is available. Please enter "
                        "a factual claim to verify."
                    )

                else:

                    try:

                        with st.spinner(
                            "🤖 The five-agent pipeline is "
                            "fact-checking the extracted claim..."
                        ):

                            fact_check_result = (
                                run_claim_analysis(
                                    claim_for_fact_check.strip()
                                )
                            )

                        st.session_state[
                            "image_fact_check_report"
                        ] = fact_check_result

                    except Exception as e:

                        st.error(
                            f"Fact-checking failed: {str(e)}"
                        )

            # Keep the final report visible after Streamlit reruns.
            if (
                "image_fact_check_report"
                in st.session_state
            ):

                st.divider()

                st.subheader(
                    "📋 Fact-Check Report"
                )

                st.markdown(
                    st.session_state[
                        "image_fact_check_report"
                    ]
                )

    else:

        st.caption(
            "Upload an image to begin."
        )
