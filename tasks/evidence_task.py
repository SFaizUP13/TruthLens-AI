from crewai import Task


def create_evidence_task(agent, claim, research_result):

    return Task(
        description=f"""
        Evaluate the quality, relevance, and strength of the
        research evidence collected for the following claim.

        CLAIM:
        {claim}

        RESEARCH REPORT:
        {research_result}

        YOUR ROLE:
        You are an Evidence Analyst. Your responsibility is to
        critically evaluate the available evidence before it is
        passed to the Fact-Check Judge.

        1. SOURCE CREDIBILITY

        For each important source in the research report:
        - Identify the publisher or issuing organization.
        - Assess whether it is an official or primary source,
          a reputable secondary source, or an unverified source.
        - Consider the publisher's expertise and relevance to
          the subject.
        - Identify missing or questionable source information.
        - Do not assume a source is reliable merely because it
          appears professional or has a familiar name.

        Important:
        You are evaluating the source information provided in the
        research report. Do not claim to have independently opened
        or verified a URL unless you actually did so.

        2. EVIDENCE RELEVANCE

        Determine whether each important finding:
        - Directly supports the claim.
        - Directly contradicts the claim.
        - Provides useful background but does not establish
          whether the claim is true or false.
        - Is too weak or unclear to draw a conclusion from.

        Explain why the evidence is relevant to the claim.

        3. INDEPENDENT CORROBORATION

        Assess whether multiple sources independently support
        the same finding.

        Do not treat several articles repeating the same original
        report as independent confirmation.

        If the research report does not establish whether sources
        are independent, explicitly state this limitation.

        4. DATES AND CONTEXT

        Check the dates and context provided in the research report.

        Identify:
        - Outdated evidence.
        - Differences between event dates and publication dates.
        - Evidence referring to a different person, event, or
          time period.
        - Missing context that could change the interpretation.

        Do not invent dates or facts that are not provided.

        5. CONTRADICTORY EVIDENCE

        Compare supporting and contradicting findings.

        Identify important disagreements between sources and
        explain which evidence appears stronger based on source
        quality, relevance, directness, and available corroboration.

        Do not dismiss contradictory evidence without explanation.

        6. UNCERTAINTY AND LIMITATIONS

        Identify:
        - Missing primary sources.
        - Unverified URLs or source details.
        - Insufficient or indirect evidence.
        - Unresolved disagreements.
        - Limitations of the available research.

        Remember that failure to find evidence does not
        automatically prove that a claim is false.

        7. STRICT RULES

        - Do not conduct new web research.
        - Do not invent sources, facts, dates, or quotations.
        - Do not claim that you independently verified sources
          when you did not.
        - Do not issue a final TRUE or FALSE verdict.
        - Do not confuse source popularity with source reliability.
        - Clearly distinguish established facts from assumptions
          and unresolved questions.

        REQUIRED OUTPUT FORMAT:

        A. Claim Being Evaluated

        B. Source Credibility Assessment
        Identify important sources and assess their reliability
        using only the available information.

        C. Supporting Evidence
        Summarize relevant evidence supporting the claim.

        D. Contradicting Evidence
        Summarize relevant evidence contradicting the claim.

        E. Independent Corroboration
        Explain whether independent confirmation is established.

        F. Evidence Strength
        Explain which findings are strong, limited, indirect,
        or inconclusive, and why.

        G. Uncertainties and Limitations
        Identify missing information and unresolved questions.

        H. Summary for the Fact-Check Judge
        Provide a balanced assessment of the evidence, including
        its strongest points and major weaknesses.

        Keep the report concise, factual, and structured.
        """,

        expected_output="""
        A structured evidence assessment containing:
        1. Source credibility assessments.
        2. Evidence supporting and contradicting the claim.
        3. Evidence relevance and strength.
        4. Assessment of independent corroboration.
        5. Important dates and contextual limitations.
        6. Uncertainties and unresolved questions.
        7. A balanced summary for the Fact-Check Judge.

        Do not issue a final TRUE or FALSE verdict.
        Do not invent facts or claim unperformed verification.
        """,

        agent=agent
    )
