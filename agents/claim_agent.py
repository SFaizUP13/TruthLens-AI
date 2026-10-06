from crewai import Agent

from utils.groq_llm import GroqLLM


def create_claim_agent():

    llm = GroqLLM()

    claim_agent = Agent(

        role="Claim Analyst",

        goal="""
        Analyze a user's claim carefully and identify
        the factual statements that need verification.
        """,

        backstory="""
        You are an expert fact-checking analyst.

        You examine claims objectively.

        You never assume that a claim is true or false.

        You identify factual assertions, important
        entities, dates, locations and information
        that should be verified.
        """,

        llm=llm,

        verbose=True
    )

    return claim_agent
