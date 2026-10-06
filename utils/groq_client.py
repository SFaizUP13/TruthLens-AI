import streamlit as st
from openai import OpenAI


def get_groq_client():

    api_key = st.secrets["GROQ_API_KEY"]

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    return client
