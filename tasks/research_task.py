from crewai import Task
from datetime import date


def create_research_task(agent, claim_task):

    verification_date = date.today().strftime("%d %B %Y")

    task = Task(
        description=f"""
        Investigate the claim analysis produced by the Claim Analyst.

        IMPORTANT:
        The current verification date is {verification_date}.

        ============================================================
        TIME-AWARE RESEARCH
        ============================================================

        Resolve relative time expressions such as:

        - today
        - yesterday
        - tomorrow
        - currently
        - recently
        - this week
        - this month

        Interpret these expressions relative to the verification
        date above.

        If the claim refers to a recent or current event:

        1. Search specifically for information from the
           verification date first.

        2. Search recent reliable news from the verification date
           and preceding few days.

        3. Search official government, institutional, scientific,
           or organizational sources.

        4. Check multiple reputable sources where possible.

        5. Pay close attention to publication dates and update dates.

        6. Do not treat an old article as evidence that an event
           did or did not happen on the verification date.

        7. Older sources may be used for historical/background
           information only.

        ============================================================
        EVIDENCE CLASSIFICATION
        ============================================================

        Every important source must be classified into one of
        these categories:

        A. CURRENT EVIDENCE

        Information published or updated very recently and directly
        relevant to the event or claim being investigated.

        B. CURRENT OFFICIAL STATUS

        An official source may contain an older publication date but
        currently identifies the status of a person, organization,
        office, policy, or institution.

        For example, an official government profile may currently
        identify a person as holding an office even if the webpage
        itself was originally published earlier.

        Clearly distinguish this from a recently published news
        report.

        C. RECENT EVIDENCE

        Reliable information from the recent past that is relevant
        to the claim but is not necessarily from the verification
        date.

        D. HISTORICAL / BACKGROUND

        Older information that provides context but does not establish
        what happened on the verification date.

        ============================================================
        SOURCE VERIFICATION AND ANTI-HALLUCINATION RULES
        ============================================================

        Before using a source as evidence:

        1. Retrieve and examine the actual source content whenever
           possible.

        2. Confirm that the source directly supports the statement
           attributed to it.

        3. Never infer an article's content from its headline,
           URL, publication date, or another article.

        4. Never claim to have read an article if its content was
           not successfully retrieved.

        5. If a source cannot be retrieved or verified, label it
           "UNVERIFIED SOURCE" and explain the limitation.

        6. Do not invent quotations, article contents, publication
           dates, updates, or official announcements.

        7. Distinguish verified facts from assumptions and
           inferences.

        8. An article published before the alleged event cannot,
           by itself, establish whether that event subsequently
           occurred.

        9. An official profile listing a person in a position
           is relevant evidence of their status, but its strength
           depends on how current and reliable the information is.

        10. Do not describe an unverified source as evidence
            contradicting the claim.

        11. If current sources cannot be retrieved, clearly report
            this limitation instead of presenting the search as
            conclusive.


        ============================================================
        AVOID ABSENCE-OF-EVIDENCE ERRORS
        ============================================================

        Be careful with negative conclusions.

        The fact that no article or announcement was found does NOT
        automatically prove that the claimed event did not happen.

        Do NOT say:

        "The claim is false because no news article was found."

        Do NOT say:

        "The claim is contradicted by the absence of an announcement."

        Instead use careful wording such as:

        "No supporting evidence was found in the sources searched."

        or:

        "No current supporting evidence was identified."

        If an official source currently indicates a different status,
        report that separately as evidence.

        Example:

        "No current supporting evidence of the reported resignation
        was found. An official government source currently identifies
        the person as Prime Minister. However, absence of a resignation
        announcement alone does not conclusively prove that no
        resignation occurred."

        ============================================================
        GENERAL RESEARCH
        ============================================================

        Investigate:

        1. Whether the main claim has been reported by credible
           sources.

        2. Who reported the information.

        3. When the information was reported.

        4. What factual evidence supports the claim.

        5. Whether credible sources disagree with the claim.

        6. Whether official sources provide relevant current status.

        7. Whether the available evidence is sufficient to support
           the claim.

        Give priority to:

        - Official government sources
        - Universities
        - Scientific publications
        - Official organizations
        - Reputable news organizations

        Compare multiple reliable sources whenever possible.

        Clearly distinguish between:

        - Current evidence
        - Current official status
        - Recent evidence
        - Historical/background information
        - Evidence supporting the claim
        - Evidence contradicting the claim
        - Missing evidence
        - Important uncertainties

        Do not make a final TRUE/FALSE verdict.
        The final verdict will be produced by another agent.

        ============================================================
        IMPORTANT
        ============================================================

        Do not invent sources.

        Do not invent publication dates.

        Do not assume that an old source is current merely because
        it contains information that may still be valid.

        Do not treat Wikipedia alone as authoritative evidence when
        an official source is available.

        Do not conclude that a claim is false solely because no
        supporting source was found.

        Do not provide the final TRUE/FALSE verdict.
        """,

        expected_output="""
        Produce a structured research report containing:

        1. Claim investigated

        2. Verification date

        3. Current evidence
           - Sources published or updated very recently
           - Directly relevant to the claim

        4. Current official status
           - Current information from official sources
           - Clearly identify the source and its date if available

        5. Recent evidence
           - Reliable recent information that is not from the
             verification date

        6. Evidence supporting the claim

        7. Evidence contradicting or challenging the claim

        8. Historical/background information

        9. Important findings

        10. Sources consulted
            - Source name
            - URL when available
            - Publication/update date when available
            - Evidence classification

        11. Key dates

        12. Important uncertainties

        13. Research conclusion

        For time-sensitive claims, explicitly state:

        - Was current evidence found?
        - Was current official status found?
        - Was recent evidence found?
        - Was only historical/background evidence found?

        - State whether the source content was actually retrieved.
        - Identify unverified sources separately.
        - Explain when evidence is indirect or outdated.
        - Never infer source content that was not examined.

        If no current supporting evidence was found, say:

        "No current supporting evidence was found."

        Do not convert this statement into a definitive FALSE
        conclusion.

        Do not provide a final TRUE/FALSE verdict.
        """,

        agent=agent,

        # Use the output of the Claim Analyst as context
        context=[claim_task]
    )

    return task
