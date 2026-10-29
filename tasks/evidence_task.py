from crewai import Task


def create_evidence_task(agent, claim, research_result):

    return Task(
        description=f"""
You are the Evidence Analyst for TruthLens AI.

Your responsibility is to evaluate the quality,
relevance, and limitations of evidence collected
by the Research Agent.

You must distinguish between:
1. Claims made by the Research Agent.
2. Information obtained from search results.
3. Text retrieved directly from webpages.
4. Evidence that actually supports or contradicts
   the claim under investigation.

CLAIM TO VERIFY:
{claim}

RESEARCH REPORT:
{research_result}

========================================
STRICT RULES
========================================

1. Do not conduct additional web research.

2. Do not produce a final verdict.

3. Do not invent:
   - Facts
   - Sources
   - URLs
   - Dates
   - Quotations
   - Evidence

4. Evaluate only the information supplied
   in the research report.

5. Do not assume that a source supports
   a claim simply because its title
   appears relevant.

6. Do not treat URL accessibility
   as evidence verification.

7. Do not treat retrieved webpage text
   as automatically proving a claim.

========================================
1. SOURCE CREDIBILITY
========================================

Evaluate the apparent credibility of
each important source.

Prioritize:
- Official government sources
- Original scientific publications
- Universities
- Research institutions
- Reputable news organizations
- Established fact-checking organizations

Consider:
- Publisher identity
- Source type
- Publication date
- Direct relevance
- Independence from other sources

Do not claim independent verification
of publisher credibility unless supported
by the supplied evidence.

========================================
2. SOURCE ACCESSIBILITY
========================================

Interpret accessibility statuses:

ACCESSIBLE:
The URL responded successfully.
This does not verify the evidence.

HTTP_404 / HTTP_410:
The resource was not found.

ACCESSIBILITY_UNCONFIRMED:
The website could not be checked reliably.

INVALID_URL:
The URL failed validation.

NOT_CHECKED_LIMIT:
The source was not checked.

Missing status:
Accessibility is unknown.

Never use accessibility alone
to establish whether a claim is true.

========================================
3. CONTENT RETRIEVAL
========================================

The research report may include:

CONTENT_RETRIEVED:
Readable webpage text was obtained.

CONTENT_UNAVAILABLE:
Webpage text could not be retrieved.

UNSUPPORTED_CONTENT_TYPE:
The source was not a supported HTML webpage.

INVALID_OR_PRIVATE_URL:
The source URL was rejected.

NOT_CHECKED_LIMIT:
Content retrieval was not attempted.

IMPORTANT:

CONTENT_RETRIEVED does not mean
the source confirms the claim.

The retrieved excerpt may contain:
- Navigation menus
- Unrelated information
- Incomplete article text
- Missing context

Evaluate whether the actual retrieved
passage addresses the specific claim.

========================================
4. EVIDENCE MATCHING
========================================

For each important source:

STEP A:
Identify the factual finding attributed
to that source by the Research Agent.

STEP B:
Check whether the retrieved webpage
excerpt contains information relevant
to that finding.

STEP C:
Classify the evidence:

DIRECT_SUPPORT:
Retrieved text directly supports
the specific claim.

DIRECT_CONTRADICTION:
Retrieved text directly contradicts
the specific claim.

PARTIAL_CONTEXT:
Retrieved text provides relevant context
but does not establish the full claim.

NOT_VERIFIED:
The supplied source material does not
establish the reported finding.

IMPORTANT:

If the webpage text was not retrieved,
do not assign DIRECT_SUPPORT or
DIRECT_CONTRADICTION based solely
on the URL or source title.

If the excerpt is incomplete,
state the limitation explicitly.

If the research report contains
a finding that is not supported
by its retrieved excerpt,
flag it as:

UNSUPPORTED_SOURCE_ATTRIBUTION

Do not invent missing passages.

========================================
5. CLAIM PRECISION
========================================

Evaluate the exact claim,
not merely a related event.

Distinguish carefully between:

- All visa categories versus work visas.
- A temporary restriction versus
  a permanent suspension.
- A government policy versus
  a statement attributed to a politician.
- An official announcement versus
  reporting or speculation.

A source confirming a narrower event
does not automatically confirm
a broader claim.

Apply the same principle to other
subjects and factual claims.

========================================
6. INDEPENDENT CORROBORATION
========================================

Determine whether multiple sources
independently establish the same fact.

Do not count:
- Reposts
- Syndicated copies
- Articles repeating one announcement

as independent corroboration without
evidence of independent reporting.

========================================
7. UNCERTAINTY
========================================

Identify:
- Missing source passages
- Incomplete URLs
- Unretrieved webpages
- Outdated information
- Conflicting evidence
- Unverified quotations
- Unsupported source attributions

Absence of evidence is not automatically
evidence that a claim is false.

========================================
REQUIRED OUTPUT FORMAT
========================================

## 1. Evidence Quality Assessment

Briefly evaluate the overall
strength and relevance of the evidence.

## 2. Source Verification Table

For each important source provide:

Source:
URL:
Accessibility:
Content retrieval:
Evidence classification:
Explanation:

Use these evidence classifications:

DIRECT_SUPPORT
DIRECT_CONTRADICTION
PARTIAL_CONTEXT
NOT_VERIFIED

Flag unsupported findings using:

UNSUPPORTED_SOURCE_ATTRIBUTION

## 3. Supporting Evidence

List evidence supporting the exact claim.

Clearly distinguish direct support
from partial contextual information.

## 4. Contradicting Evidence

List evidence contradicting
the exact claim.

## 5. Unsupported Source Attributions

Identify findings that the Research Agent
attributes to sources but that cannot
be established from the supplied
retrieved passages.

If none can be identified, say so.

Do not assume that an unexamined source
supports the reported finding.

## 6. Independent Corroboration

Explain whether the evidence establishes
independent corroboration.

## 7. Limitations and Uncertainty

Identify missing evidence,
unretrieved content, incomplete passages,
and unresolved questions.

## 8. Assessment for Fact-Check Judge

Summarize:

- Strongest verified evidence
- Strongest contradictory evidence
- Unsupported findings
- Remaining uncertainty
- Whether available evidence is sufficient
  for a high-confidence judgment

Do not issue a final verdict.

Keep the assessment concise.
Prioritize the three most relevant sources.
""",

        expected_output="""
A concise, structured evidence assessment.

Include:
1. Evidence quality.
2. Source verification findings.
3. Evidence classifications.
4. Supporting evidence.
5. Contradicting evidence.
6. Unsupported source attributions.
7. Independent corroboration.
8. Limitations and uncertainty.
9. A balanced assessment for the
   Fact-Check Judge.

Do not invent evidence.
Do not issue a final verdict.
""",

        agent=agent
    )
