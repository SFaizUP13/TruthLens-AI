from crewai import Task


def create_research_task(agent, claim):

    return Task(
        description=f"""
        Investigate the following claim using web research.

        CLAIM:
        {claim}

        YOUR OBJECTIVE:
        Find reliable, relevant evidence that helps determine
        whether this claim is accurate.

        RESEARCH RULES:

        1. SOURCE QUALITY
        - Prioritize official government websites and original
          statements from relevant authorities.
        - For scientific claims, prioritize original research
          papers and reputable scientific institutions.
        - For news claims, prioritize reputable news organizations
          and independently corroborated reporting.
        - Prefer primary sources over articles that merely repeat
          another publication.

        2. SOURCE TRACEABILITY
        For every important source, provide:
        - Exact source title.
        - Direct URL, when available.
        - Publisher or issuing organization.
        - Publication date, if available.
        - Relevant findings that relate directly to the claim.

        Never invent a source, URL, publication date, quotation,
        or piece of evidence.

        3. EVIDENCE
        Identify:
        - Evidence supporting the claim.
        - Evidence contradicting the claim.
        - Important facts that remain uncertain.
        - Whether the evidence is direct or indirect.

        Explain how each important finding relates to the claim.
        Do not treat an unrelated article as proof.

        4. CURRENT EVENTS
        For claims involving recent events:
        - Check dates carefully.
        - Distinguish the date of an event from the date an article
          was published.
        - Prefer recent, authoritative reporting.
        - Check official statements when relevant.
        - Clearly state when available evidence is insufficient
          to confirm whether the event occurred.

        5. MISSING OR CONFLICTING EVIDENCE
        If reliable sources do not establish whether the claim is
        true or false, say so explicitly.

        Absence of evidence is not automatically evidence that
        the claim is false.

        If sources disagree, explain the disagreement and identify
        which sources provide the strongest evidence.

        6. STRICT LIMITATIONS
        - Do not issue a TRUE or FALSE verdict.
        - Do not make a final credibility judgment.
        - Do not claim that you verified a source if you did not.
        - Do not present search snippets as conclusive proof.
        - Do not invent citations to make the report appear complete.

        REQUIRED OUTPUT FORMAT:

        A. Claim Under Investigation
        Restate the claim briefly.

        B. Key Findings
        Summarize the most relevant findings.

        C. Supporting Evidence
        List evidence supporting the claim, if found.

        D. Contradicting Evidence
        List evidence contradicting the claim, if found.

        E. Sources
        For each source, provide its title, publisher, direct URL,
        publication date if available, and relevant evidence.

        F. Uncertainties and Limitations
        Explain missing evidence, conflicting information,
        search limitations, or unresolved questions.

        G. Research Summary
        Briefly summarize what the available evidence establishes
        and what still requires verification.

        Keep the report factual, concise, and easy for the
        Evidence Analyst to evaluate.
        """,

        expected_output="""
        A structured research report containing:
        1. Key findings about the claim.
        2. Supporting and contradicting evidence.
        3. Source titles, publishers, URLs, and dates when available.
        4. An explanation of the relevance of each important source.
        5. Uncertainties and limitations.
        6. A neutral research summary without a TRUE or FALSE verdict.

        Sources and details must not be fabricated.
        """,

        agent=agent
    )
