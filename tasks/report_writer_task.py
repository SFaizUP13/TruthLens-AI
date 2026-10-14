from crewai import Task


def create_report_writer_task(
    agent,
    claim,
    claim_analysis,
    research_report,
    evidence_assessment,
    judge_verdict
):

    task = Task(
        description=f"""
        Prepare a user-friendly fact-checking report using
        the information provided below.

        ORIGINAL CLAIM:
        {claim}

        CLAIM ANALYSIS:
        {claim_analysis}

        RESEARCH REPORT:
        {research_report}

        EVIDENCE ASSESSMENT:
        {evidence_assessment}

        FACT-CHECK JUDGE'S VERDICT:
        {judge_verdict}

        INSTRUCTIONS:

        1. Present the original claim accurately.

        2. Use the verdict issued by the Fact-Check Judge.
           Do not change or independently reinterpret it.

        3. Explain the verdict in simple, understandable language.

        4. Summarize the strongest supporting and contradicting
           evidence found in the provided reports.

        5. Include relevant source names and URLs that already
           appear in the research report.

        6. Clearly explain important uncertainties and limitations.

        7. Never invent facts, quotations, sources, URLs,
           publication dates, or evidence.

        8. If the reports disagree or contain insufficient
           evidence, explain that limitation honestly.

        9. Keep the report concise, objective, and neutral.

        10. Do not conduct additional research or issue a
            new verdict.
        """,

        expected_output="""
        Produce a concise fact-checking report using these sections:

        1. CLAIM
           State the claim being checked.

        2. VERDICT
           Present the Fact-Check Judge's verdict exactly as given.

        3. EXPLANATION
           Explain why the verdict was reached.

        4. KEY EVIDENCE
           Summarize the most relevant supporting and contradicting
           evidence available in the provided reports.

        5. SOURCES
           List relevant source names and URLs from the research
           report. Do not invent or modify URLs.

        6. LIMITATIONS
           Identify important uncertainties, missing evidence,
           or source limitations.

        7. FINAL SUMMARY
           Provide a brief conclusion consistent with the verdict.

        Use clear headings and plain language.
        Do not add unsupported information.
        """,

        agent=agent
    )

    return task
