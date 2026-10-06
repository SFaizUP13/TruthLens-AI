from crewai import Task


def create_research_task(agent, claim_analysis):

    task = Task(
        description=f"""
        Investigate the following claim analysis:

        {claim_analysis}

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

        Give priority to official organizations,
        universities, scientific publications and
        reputable news organizations.

        Do not make a final TRUE/FALSE verdict.
        The final verdict will be produced by another agent.
        """,

        expected_output="""
        Produce a structured research report containing:

        - Claim investigated
        - Evidence supporting the claim
        - Evidence contradicting the claim
        - Important findings
        - Sources consulted
        - Key dates
        - Important uncertainties
        - Research conclusion

        Include source names and URLs whenever available.

        Do not invent sources.
        Do not provide a final TRUE/FALSE verdict.
        """,

        agent=agent
    )

    return task
