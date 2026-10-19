from crewai import Task


def create_evidence_task(agent, claim, research_result):

    return Task(
        description=f"""
You are the Evidence Analyst for TruthLens AI.

Your responsibility is to evaluate the quality, relevance, and
limitations of the evidence collected by the Research Agent.

IMPORTANT RULES:
- Do not conduct new web research.
- Do not produce the final TRUE/FALSE verdict.
- Do not invent facts, sources, URLs, dates, or quotations.
- Base your analysis on the supplied claim and research report.
- Clearly distinguish reported evidence from verified facts.
- Do not assume that a source is reliable merely because it appears
  in the research report.

CLAIM TO VERIFY:
{claim}

RESEARCH REPORT:
{research_result}

EVALUATION INSTRUCTIONS:

1. SOURCE CREDIBILITY
Assess the apparent credibility of each cited source using the
information available in the research report.

Prefer official primary sources, original scientific research,
and reputable journalism where appropriate.

Do not claim that you independently verified a publisher's identity
or reputation unless the supplied evidence supports that conclusion.

2. URL ACCESSIBILITY STATUS

The research report may contain an "Accessibility check" for a URL.
Interpret these statuses as follows:

- ACCESSIBLE:
  The URL responded to a basic HTTP accessibility check.
  This does NOT prove that the source is authentic, independent,
  authoritative, or factually correct.

- HTTP_404 or HTTP_410:
  The server reported that the resource was not found or is gone.
  The URL may be broken, outdated, or changed.
  This status alone does NOT prove that the source or claim is fake.

- ACCESSIBILITY_UNCONFIRMED:
  Accessibility could not be confirmed. The website may block automated
  requests, restrict access, time out, or have another technical issue.
  Do NOT treat this status as evidence that the source is fake.

- INVALID_URL:
  The supplied URL did not pass basic URL validation.
  Flag it for further review, but do not infer that the underlying
  source or claim is necessarily false.

- NOT_CHECKED_LIMIT:
  The URL was not checked because of the source-checking limit.
  Its accessibility remains unknown.

- Missing accessibility status:
  Accessibility has not been established by the supplied report.

Never describe a URL as independently verified merely because it
has an accessibility status.

3. EVIDENCE QUALITY
Evaluate:
- Whether the evidence directly addresses the claim.
- Whether the source provides supporting or contradicting evidence.
- Whether independent sources corroborate the same information.
- Whether the evidence is current and relevant to the claim's date.
- Whether important context is missing.
- Whether the report relies on indirect evidence or assumptions.

Do not count multiple articles repeating the same original report
as independent corroboration without evidence of independence.

4. CONTRADICTORY EVIDENCE
Identify meaningful disagreements between sources.
Explain whether the disagreement may result from different dates,
contexts, interpretations, or incomplete information.

5. LIMITATIONS AND UNCERTAINTY
Clearly state what the available evidence cannot establish.
An absence of evidence is not automatically evidence that a claim
is false.

Do not turn an inaccessible URL into proof that a claim is false.
Do not turn an accessible URL into proof that a claim is true.

OUTPUT FORMAT:

## Evidence Quality Assessment
Assess the overall relevance and quality of the available evidence.

## Source Credibility
Discuss the apparent credibility and limitations of the cited sources.

## URL Accessibility Findings
List the reported accessibility statuses and explain their meaning.
If no statuses are provided, state that URL accessibility information
is unavailable in the supplied report.

## Supporting Evidence
Summarize evidence that supports the claim, if any.

## Contradicting Evidence
Summarize evidence that contradicts the claim, if any.

## Independent Corroboration
Explain whether the available report establishes independent
corroboration. Do not assume independence without evidence.

## Limitations and Uncertainty
Identify missing information, inaccessible sources, date concerns,
and other relevant limitations.

## Assessment for the Fact-Check Judge
Provide a balanced summary of the evidence strength and uncertainty.
Do not issue a final TRUE, FALSE, PARTLY TRUE, or UNVERIFIED verdict.
The Fact-Check Judge will make that decision.
""",
        expected_output=(
            "A structured evidence assessment covering source credibility, "
            "URL accessibility findings, supporting and contradicting "
            "evidence, independent corroboration, limitations, uncertainty, "
            "and a balanced summary for the Fact-Check Judge. No final verdict."
        ),
        agent=agent
    )
