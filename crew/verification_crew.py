from crewai import Crew, Process

from agents.claim_agent import create_claim_agent
from agents.research_agent import create_research_agent

from tasks.claim_task import create_claim_task
from tasks.research_task import create_research_task


def run_claim_analysis(claim):

    # -------------------------
    # Agent 1: Claim Analyst
    # -------------------------

    claim_agent = create_claim_agent()

    claim_task = create_claim_task(
        claim_agent,
        claim
    )

    # -------------------------
    # Agent 2: Research Agent
    # -------------------------

    research_agent = create_research_agent()

    research_task = create_research_task(
        research_agent,
        claim_task
    )

    # -------------------------
    # Crew
    # -------------------------

    crew = Crew(
        agents=[
            claim_agent,
            research_agent
        ],

        tasks=[
            claim_task,
            research_task
        ],

        process=Process.sequential,

        verbose=True
    )

    result = crew.kickoff()

    return result
