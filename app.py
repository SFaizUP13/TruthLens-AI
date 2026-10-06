import streamlit as st

from utils.groq_test import test_groq


st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎"
)

st.title("🔎 TruthLens AI")

st.write("Testing Groq + GPT-OSS-20B")


if st.button("Test Groq"):

    try:

        result = test_groq()

        st.success("Groq connection successful!")

        st.write(result)

    except Exception as e:

        st.error(f"Error: {e}")
