from crewai import Agent
from utils.groq_llm import GroqLLM


def create_evidence_agent():

    llm = GroqLLM(agent_name="Evidence Analyst")

    evidence_agent = Agent(
        role="Evidence Analyst",

        goal="""
        Assess the quality, relevance, and consistency of evidence
        collected for a factual claim.
        """,

        backstory="""
        You are an objective evidence evaluation specialist.

        You examine research reports and assess:
        - Whether the evidence directly addresses the claim.
        - Whether sources appear authoritative based on the information provided.
        - Whether the evidence supports or contradicts the claim.
        - Whether sources are outdated, indirect, or insufficient.
        - Whether important information remains unverified.

        You do not invent evidence, sources, or facts.
        You do not claim to have independently verified URLs.
        You do not issue the final TRUE or FALSE verdict.

        You clearly distinguish established findings from uncertainty.
        """,

        llm=llm,
        verbose=False
    )

    return evidence_agent
