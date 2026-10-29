from crewai import Task


def create_research_task(agent, claim):

    return Task(
        description=f"""
        Investigate the following factual claim
        using reliable web research.

        CLAIM:
        {claim}

        OBJECTIVE:
        Find evidence that supports, contradicts,
        or provides context for this claim.

        RESEARCH RULES:

        1. SOURCE PRIORITY

        Prefer:
        - Official government sources
        - Original announcements
        - Scientific publications
        - Universities and research institutions
        - Reputable news organizations
        - Independent fact-checking organizations

        Prefer primary sources whenever possible.

        2. EVIDENCE TRACEABILITY

        For each important source, report:

        - Source title
        - Publisher
        - Publication date, if available
        - Direct URL
        - Relevant evidence
        - Evidence relationship:
          SUPPORTS / CONTRADICTS / CONTEXT / UNCLEAR
        - Evidence access:
          CONTENT_INSPECTED / SEARCH_RESULT_ONLY / UNKNOWN

        Use CONTENT_INSPECTED only when the actual
        source content was available to you.

        Use SEARCH_RESULT_ONLY when your information
        comes only from a search result or snippet.

        Use UNKNOWN when the available information
        does not establish what was inspected.

        Do not invent evidence, URLs, dates,
        publishers, or source titles.

        3. SOURCE PASSAGES

        When actual source content is available,
        provide one short, relevant quotation
        or an accurately paraphrased passage.

        Clearly label quotations and paraphrases.

        Never fabricate quotations.

        If the relevant passage is unavailable,
        write:

        Relevant passage: NOT VERIFIED

        4. EVIDENCE RELATIONSHIP

        For every source, explain whether it:

        - Supports the claim
        - Contradicts the claim
        - Provides context only
        - Is inconclusive

        Do not classify a source as supporting
        or contradicting a claim merely because
        its title appears relevant.

        5. DATE VERIFICATION

        For recent events:
        - Check publication dates.
        - Distinguish event dates from publication dates.
        - Identify outdated evidence.
        - Prefer recent official information.

        6. UNCERTAINTY

        Absence of evidence does not automatically
        establish that a claim is false.

        If evidence is missing or conflicting,
        clearly explain the limitations.

        7. STRICT RULES

        - Do not issue a final TRUE/FALSE verdict.
        - Do not invent source quotations.
        - Do not claim to have opened a source
          unless its content was actually available.
        - Do not confuse URL accessibility
          with evidence verification.
        - Do not treat search snippets
          as conclusive proof.

        REQUIRED OUTPUT:

        A. CLAIM
        Restate the claim.

        B. KEY FINDINGS
        Summarize the most relevant findings.

        C. SOURCE EVIDENCE
        For each important source provide:

        Source:
        Publisher:
        Date:
        URL:
        Relevant passage:
        Evidence relationship:
        Evidence access:
        Explanation:

        D. MISSING OR CONFLICTING EVIDENCE
        Identify unresolved questions.

        E. RESEARCH SUMMARY
        Summarize what the evidence establishes
        without issuing a final verdict.

        Keep the report concise.

        Prioritize up to three highly relevant
        sources rather than many weak sources.
        """,

        expected_output="""
        A concise, evidence-based research report.

        Include:
        - Key findings
        - Up to three relevant sources
        - Source URLs and dates when available
        - Relevant passages or NOT VERIFIED
        - Evidence relationship classifications
        - Evidence access classifications
        - Important uncertainties

        Do not fabricate evidence.
        Do not issue a final verdict.
        """,

        agent=agent
    )
