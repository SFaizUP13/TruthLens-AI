from crewai import Crew, Process

from agents.claim_agent import create_claim_agent
from agents.research_agent import create_research_agent

from tasks.claim_task import create_claim_task
from tasks.research_task import create_research_task


def run_claim_analysis(claim):

    # STEP 1: Run the Claim Analyst independently
    claim_agent = create_claim_agent()
    claim_task = create_claim_task(claim_agent, claim)

    claim_crew = Crew(
        agents=[claim_agent],
        tasks=[claim_task],
        process=Process.sequential,
        verbose=False
    )

    claim_result = claim_crew.kickoff()

    # STEP 2: Run the Research Agent in a separate crew
    # Pass the original claim, not the Claim Analyst's full report.
    research_agent = create_research_agent()
    research_task = create_research_task(research_agent, claim)

    research_crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False
    )

    research_result = research_crew.kickoff()

    # STEP 3: Return both results to the Streamlit application
    return (
        "## Claim Analysis\n\n"
        + str(claim_result)
        + "\n\n## Research Report\n\n"
        + str(research_result)
    )
