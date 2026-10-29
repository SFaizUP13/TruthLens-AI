import re

from crewai import Crew, Process

from agents.claim_agent import create_claim_agent
from agents.research_agent import create_research_agent
from agents.evidence_agent import create_evidence_agent
from agents.judge_agent import create_judge_agent
from agents.report_writer_agent import create_report_writer_agent

from tasks.claim_task import create_claim_task
from tasks.research_task import create_research_task
from tasks.evidence_task import create_evidence_task
from tasks.judge_task import create_judge_task
from tasks.report_writer_task import create_report_writer_task


# ==========================================
# CONFIGURATION
# ==========================================

VALID_VERDICTS = {
    "TRUE",
    "FALSE",
    "PARTLY TRUE",
    "UNVERIFIED",
}

VALID_CONFIDENCE = {
    "HIGH",
    "MEDIUM",
    "LOW",
}


# ==========================================
# CREW HELPER
# ==========================================

def run_single_agent(agent, task):
    """
    Run one CrewAI agent and return its text.
    """

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


# ==========================================
# EXTRACT JUDGE DECISION
# ==========================================

def extract_field(text, field_name):
    """
    Extract a field such as:

    VERDICT: FALSE
    CONFIDENCE: HIGH

    Also supports basic Markdown formatting.
    """

    pattern = (
        rf"(?im)^\s*"
        rf"(?:\#{1,6}\s*)?"
        rf"(?:\*\*)?"
        rf"{re.escape(field_name)}"
        rf"(?:\*\*)?"
        rf"\s*:\s*"
        rf"(?:\*\*)?"
        rf"([^\n*]+)"
    )

    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    return None


def extract_judge_decision(judge_text):
    """
    Read the verdict and confidence
    returned by the Fact-Check Judge.
    """

    verdict_text = extract_field(
        judge_text,
        "VERDICT",
    )

    confidence_text = extract_field(
        judge_text,
        "CONFIDENCE",
    )

    verdict = (
        verdict_text.upper().strip()
        if verdict_text
        else "UNVERIFIED"
    )

    confidence = (
        confidence_text.upper().strip()
        if confidence_text
        else "LOW"
    )

    if verdict not in VALID_VERDICTS:
        verdict = "UNVERIFIED"

    if confidence not in VALID_CONFIDENCE:
        confidence = "LOW"

    return verdict, confidence


# ==========================================
# SOURCE EVIDENCE CHECK
# ==========================================

def assess_source_verification(research_text):
    """
    Conservatively inspect source retrieval
    information supplied by GroqResearchLLM.

    IMPORTANT:
    This checks whether source text was
    retrieved, not whether a source
    semantically proves the claim.
    """

    retrieved_count = len(
        re.findall(
            r"Content retrieval:\s*CONTENT_RETRIEVED",
            research_text,
            flags=re.IGNORECASE,
        )
    )

    unavailable_count = len(
        re.findall(
            r"Content retrieval:\s*"
            r"(?:CONTENT_UNAVAILABLE|"
            r"UNSUPPORTED_CONTENT_TYPE|"
            r"INVALID_OR_PRIVATE_URL)",
            research_text,
            flags=re.IGNORECASE,
        )
    )

    return {
        "retrieved_count": retrieved_count,
        "unavailable_count": unavailable_count,
    }


# ==========================================
# CONFIDENCE VALIDATION
# ==========================================

def validate_confidence(
    verdict,
    confidence,
    research_text,
    evidence_text,
):
    """
    Apply conservative confidence safeguards.

    A successful webpage retrieval is not
    sufficient to establish verified evidence.
    """

    source_status = assess_source_verification(
        research_text
    )

    retrieved_count = source_status[
        "retrieved_count"
    ]

    # We do not have a machine-verifiable
    # claim-to-passage evidence match yet.
    #
    # Therefore HIGH confidence is not
    # permitted at this stage.

    if confidence == "HIGH":
        confidence = "MEDIUM"

    # If no webpage content was retrieved,
    # the conclusion must be low confidence.

    if retrieved_count == 0:
        confidence = "LOW"

    # A verdict of UNVERIFIED should not
    # imply certainty about the claim.

    if verdict == "UNVERIFIED":
        confidence = "LOW"

    return confidence, source_status


# ==========================================
# CLEAN REPORT WRITER OUTPUT
# ==========================================

def clean_report(report_text):
    """
    Remove repeated TruthLens AI headings.

    Preserve the rest of the report.
    """

    report_text = re.sub(
        r"(?im)^\s*\#{1,6}\s*"
        r"TruthLens AI\s*[—–-]\s*"
        r"Fact-Check Report\s*$",
        "",
        report_text,
    )

    return report_text.strip()


# ==========================================
# MAIN FACT-CHECK PIPELINE
# ==========================================

def run_claim_analysis(claim):

    # --------------------------------------
    # STEP 1: CLAIM ANALYST
    # --------------------------------------

    claim_agent = create_claim_agent()

    claim_task = create_claim_task(
        claim_agent,
        claim,
    )

    claim_result = run_single_agent(
        claim_agent,
        claim_task,
    )

    # --------------------------------------
    # STEP 2: RESEARCH AGENT
    # --------------------------------------

    research_agent = create_research_agent()

    research_task = create_research_task(
        research_agent,
        claim,
    )

    research_result = run_single_agent(
        research_agent,
        research_task,
    )

    # --------------------------------------
    # STEP 3: EVIDENCE ANALYST
    # --------------------------------------

    evidence_agent = create_evidence_agent()

    evidence_task = create_evidence_task(
        evidence_agent,
        claim,
        research_result,
    )

    evidence_result = run_single_agent(
        evidence_agent,
        evidence_task,
    )

    # --------------------------------------
    # STEP 4: FACT-CHECK JUDGE
    # --------------------------------------

    judge_agent = create_judge_agent()

    judge_task = create_judge_task(
        judge_agent,
        claim,
        research_result,
        evidence_result,
    )

    judge_result = run_single_agent(
        judge_agent,
        judge_task,
    )

    # --------------------------------------
    # STEP 5: VALIDATE JUDGE DECISION
    # --------------------------------------

    verdict, confidence = extract_judge_decision(
        judge_result
    )

    confidence, source_status = validate_confidence(
        verdict,
        confidence,
        research_result,
        evidence_result,
    )

    print(
        "[FACT-CHECK VALIDATION] "
        f"verdict={verdict}, "
        f"confidence={confidence}, "
        f"retrieved_sources="
        f"{source_status['retrieved_count']}"
    )

    # Give the Report Writer the validated
    # verdict and confidence, not just
    # the original Judge output.

    validated_judge_result = (
        f"VALIDATED VERDICT: {verdict}\n"
        f"VALIDATED CONFIDENCE: {confidence}\n\n"
        f"ORIGINAL JUDGE ASSESSMENT:\n"
        f"{judge_result}\n\n"
        "IMPORTANT: The validated verdict "
        "and confidence override the original "
        "Judge output. Do not change them."
    )

    # --------------------------------------
    # STEP 6: REPORT WRITER
    # --------------------------------------

    report_writer = create_report_writer_agent()

    report_task = create_report_writer_task(
        report_writer,
        claim,
        claim_result,
        research_result,
        evidence_result,
        validated_judge_result,
    )

    final_report = run_single_agent(
        report_writer,
        report_task,
    )

    final_report = clean_report(
        final_report
    )

    # --------------------------------------
    # STEP 7: RETURN FINAL REPORT
    # --------------------------------------

    # Remove any verdict/confidence sections
    # generated by the Report Writer so that
    # the displayed values are controlled
    # by our Python validation.

    final_report = re.sub(
        r"(?ims)^\s*\#{1,6}\s*"
        r"(?:2\.\s*)?Verdict\s*\n"
        r".*?"
        r"(?=^\s*\#{1,6}\s|\Z)",
        "",
        final_report,
    )

    final_report = re.sub(
        r"(?ims)^\s*\#{1,6}\s*"
        r"(?:3\.\s*)?Confidence\s*\n"
        r".*?"
        r"(?=^\s*\#{1,6}\s|\Z)",
        "",
        final_report,
    )

    return (
        "## TruthLens AI — Fact-Check Report\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"**Confidence:** {confidence}\n\n"
        + final_report.strip()
    )
