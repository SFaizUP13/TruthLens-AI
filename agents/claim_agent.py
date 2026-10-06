import streamlit as st
from crewai import Agent, LLM


def create_claim_agent():

    # Get Groq API key from Streamlit secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Connect CrewAI to Groq's OpenAI-compatible API
    llm = LLM(
        model="openai/gpt-oss-20b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )

    # Create the Claim Analyst agent
    claim_agent = Agent(

        role="Claim Analyst",

        goal="""
        Analyze a user's claim carefully and identify
        the factual statements that need verification.
        """,

        backstory="""
        You are an expert fact-checking analyst.

        You examine claims objectively.

        You do not assume that a claim is true or false.

        You identify factual assertions, important
        entities, dates, locations and information
        that should be verified.
        """,

        llm=llm,

        verbose=True
    )

    return claim_agent
