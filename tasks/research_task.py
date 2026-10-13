from crewai import Task
from datetime import date


def create_research_task(agent, claim_task, claim):

    verification_date = date.today().strftime("%d %B %Y")

    task = Task(
        description=f"""
        Investigate the claim analysis provided by the Claim Analyst.

        Verification date: {verification_date}

        RESEARCH:
        1. Investigate the specific claim and relevant dates.
        2. Search for current information when the claim concerns
           a recent event.
        3. Prioritize official sources and reputable news outlets.
        4. Compare sources when possible.
        5. Examine the actual source content before relying on it.

        CLASSIFY EVIDENCE:
        - CURRENT: Published or updated recently and directly
          relevant to the event.
        - CURRENT OFFICIAL STATUS: An official source indicating
          a person's or organization's present status.
        - RECENT: Relevant information from the recent past.
        - HISTORICAL: Older information useful only as background.
        - UNVERIFIED: A source whose relevant content could not
          be retrieved or confirmed.

        SOURCE RULES:
        - Include source names, URLs, and dates when available.
        - Never invent sources, quotations, dates, or findings.
        - Never claim to have read content that was not retrieved.
        - Do not infer article contents from headlines or URLs.
        - An older article cannot establish what happened later.
        - Clearly identify indirect, outdated, or unverified evidence.

        CONCLUSION RULES:
        - Distinguish supporting evidence from contradicting evidence.
        - Absence of news is not proof that an event did not happen.
        - An outdated official profile may not reflect a recent change.
        - If current evidence cannot be found, state that limitation.
        - Do not make the final TRUE/FALSE verdict.

        Keep the report concise. Prioritize evidence quality
        over the number of sources.
        """,

        expected_output="""
        Produce a concise research report containing:

        1. Claim and verification date.
        2. Current evidence.
        3. Current official status, if available.
        4. Recent and historical evidence.
        5. Evidence supporting the claim.
        6. Evidence contradicting or challenging the claim.
        7. Sources, URLs, dates, and verification status.
        8. Important uncertainties.
        9. Research conclusion without a final verdict.

        Explicitly state whether current supporting evidence
        was found.

        Never present an unverified source as confirmed evidence.
        Never conclude that a claim is false solely because
        no supporting source was found.

        Use concise bullet points and avoid repeating findings.
        """,

        agent=agent,
        context=[claim_task]
    )

    return task
