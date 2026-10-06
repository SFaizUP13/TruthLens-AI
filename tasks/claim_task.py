from crewai import Task


def create_claim_task(agent, claim):

    task = Task(

        description=f"""
        Analyze the following claim:

        "{claim}"

        Identify:

        1. The main factual claim.
        2. Any people, organizations or places mentioned.
        3. Any dates or time references.
        4. Specific facts that need verification.
        5. What evidence should be collected.

        Do not determine whether the claim is true or
        false yet.

        Your task is only to prepare the claim for
        further investigation.
        """,

        expected_output="""
        Provide a structured analysis containing:

        - Main claim
        - Key entities
        - Dates
        - Locations
        - Verification questions
        - Required evidence

        Do not invent information.
        """,

        agent=agent
    )

    return task
