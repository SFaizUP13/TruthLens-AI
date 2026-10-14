from crewai import Crew, Process

from agents.claim_agent import create_claim_agent
from agents.research_agent import create_research_agent
from agents.evidence_agent import create_evidence_agent

from tasks.claim_task import create_claim_task
from tasks.research_task import create_research_task
from tasks.evidence_task import create_evidence_task


def run_claim_analysis(claim):

    # STEP 1: Analyze the original claim
    claim_agent = create_claim_agent()
    claim_task = create_claim_task(claim_agent, claim)

    claim_crew = Crew(
        agents=[claim_agent],
        tasks=[claim_task],
        process=Process.sequential,
        verbose=False
    )

    claim_result = claim_crew.kickoff()

    # STEP 2: Research the original claim
    research_agent = create_research_agent()
    research_task = create_research_task(research_agent, claim)

    research_crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False
    )

    research_result = research_crew.kickoff()

    # STEP 3: Assess the collected evidence
    evidence_agent = create_evidence_agent()

    evidence_task = create_evidence_task(
        evidence_agent,
        claim,
        str(research_result)
    )

    evidence_crew = Crew(
        agents=[evidence_agent],
        tasks=[evidence_task],
        process=Process.sequential,
        verbose=False
    )

    evidence_result = evidence_crew.kickoff()

    # STEP 4: Return the results to the Streamlit application
    return (
        "## Claim Analysis\n\n"
        + str(claim_result)
        + "\n\n## Research Report\n\n"
        + str(research_result)
        + "\n\n## Evidence Assessment\n\n"
        + str(evidence_result)
    )
