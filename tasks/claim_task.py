from crewai import Task


def create_claim_task(agent, claim):

    task = Task(
        description=f"""
        Analyze this claim for fact-checking:

        "{claim}"

        Identify:
        1. The main factual claim.
        2. People, organizations, and places.
        3. Relevant dates and time expressions.
        4. Questions that need verification.
        5. Evidence needed to verify the claim.

        Do not decide whether the claim is true or false.
        Do not invent missing details.

        Keep the analysis concise and factual.
        """,

        expected_output="""
        Provide a concise report with:
        - Main claim
        - Key entities
        - Dates and locations
        - Verification questions
        - Required evidence

        Use short bullet points.
        Do not give a final verdict.
        """,

        agent=agent
    )

    return task
