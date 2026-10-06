import streamlit as st
from utils.groq_client import get_groq_client


st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎"
)

st.title("🔎 TruthLens AI")

st.write(
    "AI-powered misinformation and media verification system"
)

claim = st.text_area(
    "Enter a news claim",
    placeholder="Example: Scientists have discovered a new planet..."
)


if st.button("Verify"):

    if not claim:
        st.warning("Please enter a claim.")
        st.stop()

    try:

        client = get_groq_client()

        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=f"""
You are a fact-checking assistant.

Analyze the following claim:

{claim}

Explain:
1. What the claim says
2. What would need to be verified
3. Whether the claim appears suspicious
4. What evidence should be searched for

Do not invent evidence.
"""
        )

        st.subheader("Initial Analysis")

        st.write(response.output_text)

    except Exception as e:

        st.error(f"Error: {e}")
