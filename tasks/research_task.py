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

        ============================================================
        SPECIAL RULE FOR TIME-SENSITIVE CLAIMS
        ============================================================

        If the claim refers to something happening "today",
        "yesterday", "currently", "recently", or another recent
        time period:

        1. Search specifically for information from the
           verification date first.

        2. Search recent news from the verification date and
           the preceding few days.

        3. Search official government, institutional, or
           organizational sources for the same period.

        4. Check multiple reputable news organizations.

        5. Pay close attention to the publication date of
           every source.

        6. Do NOT use an old article as evidence that an event
           did or did not happen on the current verification date.

        7. Older sources may be used only as historical
           background or context.

        8. Clearly distinguish between:
           - Current evidence
           - Recent evidence
           - Historical/background information

        9. If current evidence cannot be found, explicitly say:
           "No current evidence was found."

        10. Do not conclude that an event did not happen merely
            because older sources do not mention it.

        ============================================================
        GENERAL RESEARCH
        ============================================================

        Conduct web research to find reliable evidence
        relevant to the claim.

        Research the following:

        1. Whether the main claim has been reported by
           credible sources.

        2. Who reported the information.

        3. When the information was reported.

        4. What factual evidence supports the claim.

        5. Whether credible sources disagree with the claim.

        6. Whether the evidence is sufficient to support
           the claim.

        Search for multiple reliable sources.

        Give priority to:
        - Official government organizations
        - Universities
        - Scientific publications
        - Official organizations
        - Reputable news organizations

        Compare information from multiple sources whenever
        possible.

        Clearly distinguish between:
        - Current evidence
        - Recent evidence
        - Historical/background information
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
        - Current/recent evidence
        - Evidence supporting the claim
        - Evidence contradicting the claim
        - Historical/background information
        - Important findings
        - Sources consulted
        - Publication dates of important sources
        - Key dates
        - Important uncertainties
        - Research conclusion

        For time-sensitive claims, clearly state:

        1. Whether current evidence was found.
        2. Whether recent evidence was found.
        3. Whether only historical evidence was found.

        Do not use historical sources as evidence of what
        happened on the verification date.

        Include source names and URLs whenever available.

        Do not invent sources.

        Do not provide a final TRUE/FALSE verdict.
        """,

        agent=agent,

        # Use the output of the Claim Analyst as context
        context=[claim_task]
    )

    return task
