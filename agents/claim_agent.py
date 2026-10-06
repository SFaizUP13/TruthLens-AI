from crewai import Agent, LLM
import streamlit as st


def create_claim_agent():

    # Get Groq API key from Streamlit secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Configure CrewAI to use Groq
    llm = LLM(
        model="groq/openai/gpt-oss-20b",
        api_key=groq_api_key
    )

    # Create the agent
    claim_agent = Agent(

        role="Claim Analyst",

        goal="""
        Analyze a user's claim carefully and identify
        the factual statements that need verification.
        """,

        backstory="""
        You are an expert fact-checking analyst.

        Your job is to examine claims objectively.

        You do not assume that a claim is true or false.

        You identify factual assertions, important
        entities, dates, locations and information
        that should be verified.
        """,

        llm=llm,

        verbose=True
    )

    return claim_agent
