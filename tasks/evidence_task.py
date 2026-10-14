from crewai import Task


def create_evidence_task(agent, claim, research_report):

    task = Task(
        description=f"""
        Assess the evidence collected for the following claim.

        ORIGINAL CLAIM:
        {claim}

        RESEARCH REPORT:
        {research_report}

        Instructions:

        1. Identify the strongest evidence relevant to the claim.
        2. Separate supporting evidence from contradicting evidence.
        3. Assess source credibility using the information provided.
        4. Identify outdated, indirect, or potentially unreliable sources.
        5. Highlight missing evidence and unresolved questions.
        6. Do not invent facts, sources, URLs, or publication dates.
        7. Do not claim that you independently visited or verified sources.
        8. Do not treat an absence of evidence as proof of falsity.
        9. Do not issue a final TRUE or FALSE verdict.

        Keep the assessment concise, objective, and evidence-based.
        """,

        expected_output="""
        Return a concise evidence assessment with these sections:

        1. Strongest supporting evidence
        2. Strongest contradicting evidence
        3. Source credibility assessment
        4. Missing or uncertain evidence
        5. Overall evidence strength: Strong, Moderate, Weak, or Insufficient

        Explain the reasons for your assessment.
        Do not give a final TRUE or FALSE verdict.
        """,

        agent=agent
    )

    return task
