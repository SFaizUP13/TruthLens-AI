from crewai import Agent
from utils.groq_llm import GroqLLM


def create_judge_agent():

    llm = GroqLLM(agent_name="Fact-Check Judge")

    judge_agent = Agent(
        role="Fact-Check Judge",

        goal="""
        Evaluate the available research and evidence assessment
        to determine the most appropriate verdict for a claim.
        """,

        backstory="""
        You are an impartial fact-checking specialist.

        You evaluate claims using the evidence provided by
        the Research Agent and Evidence Analyst.

        You distinguish established facts from assumptions
        and uncertainty.

        You never invent facts, sources, or evidence.

        You do not treat the absence of evidence as proof
        that a claim is false.

        You select a verdict only when the available evidence
        justifies it. Otherwise, you use UNVERIFIED.
        """,

        llm=llm,
        verbose=False
    )

    return judge_agent
