from crewai import Task
from datetime import date


def create_research_task(agent, claim_task):

    verification_date = date.today().strftime("%d %B %Y")

    task = Task(
        description=f"""
        Investigate the claim analysis produced by the Claim Analyst.

        IMPORTANT:
        The current verification date is {verification_date}.

        Resolve relative time expressions such as:
        - today
        - yesterday
        - tomorrow
        - this week
        - currently
        - recently

        relative to the current verification date.

        If the claim contains a time-sensitive expression such as
        "today", "yesterday", "currently", or "recently", prioritize
        the most recent available evidence and explicitly check whether
        the claimed event occurred on or around the relevant date.

        Conduct web research to find reliable evidence
        relevant to the claim.

        Research the following:

        1. Whether the main claim has been reported by
           credible sources.

        2. Who reported the information.

        3. When the information was reported.

        4. What scientific or factual evidence supports
           the claim.

        5. Whether credible sources disagree with the claim.

        6. Whether the evidence is sufficient to support
           the claim.

        Search for multiple reliable sources.

        Give priority to:
        - Official government organizations
        - Universities
        - Scientific publications
        - Reputable news organizations

        For time-sensitive claims, prioritize recent sources
        over older historical sources.

        Compare information from multiple sources whenever
        possible.

        Clearly distinguish between:
        - Evidence supporting the claim
        - Evidence contradicting the claim
        - Missing evidence
        - Important uncertainties

        Do not make a final TRUE/FALSE verdict.
        The final verdict will be produced by another agent.
        """,

        expected_output="""
        Produce a structured research report containing:

        - Claim investigated
        - Verification date
        - Evidence supporting the claim
        - Evidence contradicting the claim
        - Important findings
        - Sources consulted
        - Key dates
        - Important uncertainties
        - Research conclusion

        For time-sensitive claims, clearly state whether
        current/recent evidence was found.

        Include source names and URLs whenever available.

        Do not invent sources.

        Do not provide a final TRUE/FALSE verdict.
        """,

        agent=agent,

        # Use the output of the Claim Analyst as context
        context=[claim_task]
    )

    return task
