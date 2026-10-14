from crewai import Task


def create_judge_task(
    agent,
    claim,
    research_report,
    evidence_assessment
):

    task = Task(
        description=f"""
        Evaluate the following claim using the research
        report and evidence assessment.

        ORIGINAL CLAIM:
        {claim}

        RESEARCH REPORT:
        {research_report}

        EVIDENCE ASSESSMENT:
        {evidence_assessment}

        INSTRUCTIONS:

        1. Evaluate the evidence relevant to the exact claim.
        2. Consider supporting and contradicting evidence.
        3. Consider the credibility and limitations of sources.
        4. Do not invent facts, sources, or URLs.
        5. Do not assume that a claim is false merely because
           evidence supporting it was not found.
        6. Distinguish facts from uncertainty.
        7. Do not make claims stronger than the evidence allows.
        8. Choose exactly one primary verdict:
           TRUE, FALSE, PARTLY TRUE, or UNVERIFIED.
        9. Explain the reasoning behind the verdict.
        10. Identify any important limitations.

        If the evidence is insufficient or inconclusive,
        choose UNVERIFIED.
        """,

        expected_output="""
        Return a structured fact-checking assessment with:

        1. VERDICT:
           TRUE, FALSE, PARTLY TRUE, or UNVERIFIED.

        2. CONFIDENCE:
           High, Medium, or Low.

        3. REASONING:
           A concise explanation of how the evidence
           supports the verdict.

        4. KEY EVIDENCE:
           The most important supporting or contradicting
           findings from the provided reports.

        5. LIMITATIONS:
           Missing evidence, uncertainty, or source limitations.

        Do not invent evidence or citations.
        Make the verdict consistent with the evidence.
        """,

        agent=agent
    )

    return task
