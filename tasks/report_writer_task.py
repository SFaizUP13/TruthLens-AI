from crewai import Task


def create_report_writer_task(
    agent,
    claim,
    claim_result,
    research_result,
    evidence_result,
    judge_result
):

    return Task(
        description=f"""
        Prepare the final TruthLens AI fact-check report using
        the claim analysis, research report, evidence assessment,
        and Fact-Check Judge's decision provided below.

        ORIGINAL CLAIM:
        {claim}

        CLAIM ANALYSIS:
        {claim_result}

        RESEARCH REPORT:
        {research_result}

        EVIDENCE ASSESSMENT:
        {evidence_result}

        FACT-CHECK JUDGE'S DECISION:
        {judge_result}

        YOUR ROLE:
        You are the Report Writer. Present the findings clearly
        and accurately for an ordinary reader.

        1. PRESERVE THE JUDGE'S DECISION

        Copy the judge's verdict exactly as provided.

        The permitted verdicts are:
        TRUE, FALSE, PARTLY TRUE, and UNVERIFIED.

        Preserve the confidence level assigned by the judge:
        HIGH, MEDIUM, or LOW.

        Do not change the verdict or confidence level.
        Do not make a new factual judgment.
        Do not replace an uncertain verdict with a definitive one.

        If the judge's decision is missing or does not clearly
        identify a permitted verdict, state that the verdict
        could not be reliably extracted. Do not invent one.

        2. EXPLAIN THE RESULT

        Explain the judge's reasoning in clear, concise language.

        Distinguish between:
        - Facts supported by the available evidence.
        - Findings that contradict the claim.
        - Information that remains uncertain.
        - Conclusions that cannot be established.

        Do not present an assumption as a verified fact.

        3. PRESENT KEY EVIDENCE

        Summarize the most relevant supporting and contradicting
        findings from the research and evidence reports.

        Explain briefly why each finding matters.

        Do not invent evidence or introduce new factual claims.

        4. PRESENT SOURCES

        Include relevant sources and URLs supplied in the
        research report.

        Preserve source titles and URLs accurately.

        Do not invent or reconstruct missing URLs.
        Do not claim a source has been independently verified
        unless the provided information establishes this.

        If source authenticity or accessibility remains uncertain,
        disclose that limitation.

        Do not treat several reports repeating the same original
        source as independent corroboration.

        5. EXPLAIN LIMITATIONS

        Identify important uncertainties, conflicting evidence,
        missing primary sources, and research limitations.

        If a claim is UNVERIFIED, explain that the available
        evidence is insufficient to establish whether it is true
        or false. Do not imply that it must be false.

        6. WRITE FOR THE USER

        Use straightforward language.
        Avoid unnecessary technical terminology.
        Keep the report concise but informative.
        Do not repeat the same finding in multiple sections.
        Do not include unsupported accusations or speculation.

        REQUIRED REPORT FORMAT:

        # TruthLens AI — Fact-Check Report

        ## 1. Claim
        State the original claim without changing its meaning.

        ## 2. Verdict
        Present the judge's exact verdict.

        ## 3. Confidence
        Present the judge's exact confidence level.

        ## 4. Explanation
        Summarize the judge's reasoning in accessible language.

        ## 5. Key Evidence
        Present the most important supporting and contradicting
        findings, where available.

        ## 6. Sources
        List the relevant source titles, publishers, URLs, and
        publication dates when provided.

        If no usable sources were supplied, state this clearly.

        ## 7. Limitations and Uncertainties
        Explain what remains unresolved and what limitations
        affect the assessment.

        ## 8. Final Summary
        Summarize what the available evidence establishes,
        what it does not establish, and whether further
        verification is needed.

        IMPORTANT FINAL RULES:

        - Do not change the judge's verdict or confidence.
        - Do not invent sources, URLs, dates, or facts.
        - Do not conduct new research.
        - Do not imply that missing evidence proves a claim false.
        - Do not describe sources as independently verified
          without supporting information.
        """,

        expected_output="""
        A clear, structured Markdown fact-check report containing:
        1. The original claim.
        2. The judge's exact verdict.
        3. The judge's exact confidence level.
        4. An accessible explanation of the decision.
        5. Key supporting and contradicting evidence.
        6. Relevant source titles, publishers, and URLs.
        7. Limitations and unresolved uncertainties.
        8. A balanced final summary.

        The report must preserve the judge's decision and must
        not invent facts, sources, or verification results.
        """,

        agent=agent
    )
