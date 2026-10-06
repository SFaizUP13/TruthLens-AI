import streamlit as st
from openai import OpenAI


def test_groq():

    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input="Reply with exactly: GROQ MODEL WORKS"
    )

    return response.output_text
