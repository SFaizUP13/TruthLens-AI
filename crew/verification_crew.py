from crewai import Crew, Process

from agents.claim_agent import create_claim_agent
from tasks.claim_task import create_claim_task


def run_claim_analysis(claim):

    # Create agent
    claim_agent = create_claim_agent()

    # Create task
    claim_task = create_claim_task(
        claim_agent,
        claim
    )

    # Create Crew
    crew = Crew(

        agents=[
            claim_agent
        ],

        tasks=[
            claim_task
        ],

        process=Process.sequential,

        verbose=True
    )

    # Execute Crew
    result = crew.kickoff()

    return result
