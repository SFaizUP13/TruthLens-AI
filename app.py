import streamlit as st

from utils.groq_client import get_groq_client


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
# Test input
# --------------------------------------------------

st.subheader("🧪 Groq Connection Test")

user_question = st.text_area(
    "Enter a question:",
    placeholder="What is artificial intelligence?"
)


# --------------------------------------------------
# Ask Groq
# --------------------------------------------------

if st.button("Ask Groq"):

    if not user_question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # Create Groq client
            client = get_groq_client()

            # Send request to Groq
            response = client.responses.create(

                model="openai/gpt-oss-20b",

                input=user_question
            )

            # Display response
            st.subheader("🤖 Groq Response")

            st.write(response.output_text)

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )
