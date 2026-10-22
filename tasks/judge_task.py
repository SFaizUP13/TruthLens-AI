from crewai import Task


def create_judge_task(agent, claim, research_result, evidence_result):

    return Task(
        description=f"""
        Evaluate the claim using the research findings and
        evidence assessment provided below.

        CLAIM:
        {claim}

        RESEARCH REPORT:
        {research_result}

        EVIDENCE ASSESSMENT:
        {evidence_result}

        YOUR ROLE:
        You are the Fact-Check Judge. Make a balanced decision
        based on the available evidence, not personal assumptions.

        STEP 1: IDENTIFY THE CENTRAL CLAIM

        Identify the main factual assertion.

        Consider relevant people, events, dates, locations,
        and context. Do not change the meaning of the claim.

        If the claim uses relative time expressions such as
        "today", "yesterday", or "this morning", determine
        whether the research report establishes the intended
        date and relevant timezone.

        Do not silently assume that "today" refers to the
        publication date of a historical source.

        STEP 2: EVALUATE THE EVIDENCE

        Consider:
        - Source reliability and authority.
        - Relevance to the exact claim.
        - Publication date and recency.
        - Supporting and contradicting evidence.
        - Independent corroboration.
        - Completeness of the research.
        - Missing information and unresolved uncertainty.

        Give greater weight to relevant, reliable primary
        sources and independently corroborated evidence.

        For current events, prioritize recent sources that
        directly address the specific event and date.

        Historical articles can provide background, but do
        not treat them as sufficient evidence about a new
        event merely because they discuss a similar claim.

        Do not assume that multiple articles are independent
        confirmation if they repeat the same original report.

        Do not treat an unsupported statement in the research
        report as an established fact.

        STEP 3: DISTINGUISH MISSING EVIDENCE FROM CONTRADICTION

        Absence of evidence is not automatically evidence
        that a claim is false.

        A lack of reporting alone does not prove that an event
        did not occur.

        A source that does not mention an event does not
        directly contradict the claim merely by omitting it.

        An official statement, reliable record, or other
        credible evidence that directly contradicts the
        central assertion may support a FALSE verdict.

        Consider whether the sources searched are sufficiently
        recent, relevant, authoritative, and comprehensive.

        If the research is incomplete or outdated, acknowledge
        this limitation rather than assuming the claim is false.

        STEP 4: SELECT EXACTLY ONE VERDICT

        TRUE:
        Use when sufficiently reliable evidence supports the
        central claim and no material contradiction undermines it.

        FALSE:
        Use when sufficiently reliable evidence directly
        contradicts the central claim.

        PARTLY TRUE:
        Use when important parts of the claim are accurate,
        but other material parts are inaccurate, misleading,
        or missing essential context.

        UNVERIFIED:
        Use when available evidence is insufficient, outdated,
        unreliable, indirect, or conflicting to establish
        whether the central claim is true or false.

        For breaking news, consider the timing of the claim and
        the possibility that reliable confirmation is not yet
        available.

        Do not use UNVERIFIED merely because absolute certainty
        is impossible. Assess the quality of the available
        evidence and explain any material limitations.

        STEP 5: ASSIGN CONFIDENCE

        Select exactly one confidence level:

        HIGH:
        Strong, relevant, sufficiently recent evidence from
        reliable sources provides a clear basis for the verdict.

        MEDIUM:
        Evidence generally supports the verdict, but meaningful
        limitations or unresolved questions remain.

        LOW:
        Evidence is sparse, indirect, outdated, conflicting,
        or otherwise insufficient for a strong conclusion.

        Confidence must reflect the quality and sufficiency
        of the evidence, not how plausible the claim sounds.

        If the verdict is UNVERIFIED because evidence is
        insufficient, do not express high confidence that the
        underlying claim is true or false.

        STEP 6: EXPLAIN YOUR DECISION

        Explain:
        - Why the selected verdict fits the evidence.
        - Which findings are most important.
        - Whether credible evidence directly contradicts the claim.
        - Whether the research is recent enough.
        - What uncertainties remain.
        - What additional evidence could change the verdict.

        Do not invent facts, sources, quotations, dates, or URLs.

        Do not claim that you independently checked a source
        or website when you did not.

        Do not treat a URL accessibility result as proof that
        a source is credible or that its contents support a claim.

        Do not issue multiple competing verdicts.

        REQUIRED OUTPUT FORMAT:

        VERDICT:
        Select exactly one: TRUE, FALSE, PARTLY TRUE, UNVERIFIED.

        CONFIDENCE:
        Select exactly one: HIGH, MEDIUM, LOW.

        CLAIM:
        Restate the central claim briefly.

        REASONING:
        Explain the decision using the available evidence.

        KEY EVIDENCE:
        Identify the most important supporting and contradicting
        findings. Distinguish direct contradiction from missing
        or absent evidence.

        SOURCE QUALITY:
        Briefly explain the reliability, recency, and limitations
        of the evidence sources.

        UNCERTAINTIES:
        Explain unresolved questions or missing evidence.

        WHAT COULD CHANGE THE VERDICT:
        Identify any important additional evidence needed.

        Keep your assessment clear, evidence-based, and concise.
        """,

        expected_output="""
        A structured fact-check decision containing:
        1. Exactly one verdict: TRUE, FALSE, PARTLY TRUE,
           or UNVERIFIED.
        2. Exactly one confidence level: HIGH, MEDIUM, or LOW.
        3. The central claim.
        4. Evidence-based reasoning.
        5. Key supporting and contradicting evidence.
        6. Source-quality and recency considerations.
        7. Uncertainties and limitations.
        8. Additional evidence that could change the verdict.

        Do not invent facts or references.
        Do not treat absence of evidence as proof of falsity.
        Do not treat outdated evidence as proof of a current event.
        """,

        agent=agent
    )
