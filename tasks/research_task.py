from crewai import Task
from datetime import date


def create_research_task(agent, claim):
    verification_date = date.today().strftime("%d %B %Y")

    task = Task(
        description=f"""
        Verify the following claim using web research.

        CLAIM: "{claim}"
        VERIFICATION DATE: {verification_date}

        REQUIREMENTS:
        1. Search for current evidence relevant to the claim.
        2. Prioritize official sources and reputable news outlets.
        3. Check source content before relying on it.
        4. Distinguish supporting evidence from contradicting evidence.
        5. Include source names, URLs, and publication dates when available.
        6. Identify outdated, indirect, or unverified sources.
        7. Never invent sources, quotations, dates, or findings.
        8. Do not treat missing search results as proof that a claim is false.
        9. Do not give a final TRUE/FALSE verdict.
        10. Keep the report concise and evidence-based.

        Resolve relative dates such as "today" using the verification date.
        """,

        expected_output="""
        Return a concise report with these sections:

        1. Claim and verification date
        2. Current evidence
        3. Official status, if verified
        4. Supporting evidence
        5. Contradicting evidence
        6. Sources with URLs and dates
        7. Uncertainties and limitations
        8. Research conclusion without a final verdict

        Clearly distinguish verified findings from unverified information.
        """,

        agent=agent
    )

    return task
