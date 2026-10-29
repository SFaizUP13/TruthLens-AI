import streamlit as st

from openai import OpenAI
from crewai.llms.base_llm import BaseLLM

from utils.source_verifier import (
    check_source_url,
    fetch_source_text,
)


# ==========================================
# GROQ CONFIGURATION
# ==========================================

MODEL = "openai/gpt-oss-20b"

MAX_RESEARCH_SOURCES = 3
MAX_SOURCE_EXCERPT_CHARS = 1500


def create_client():
    """Create the Groq API client."""

    return OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


# ==========================================
# MESSAGE CLEANING
# ==========================================

def clean_messages(messages):
    """Remove unnecessary CrewAI message metadata."""

    cleaned = []

    for message in messages:

        if isinstance(message, dict):

            role = message.get("role")
            content = message.get("content", "")

            if role and content:

                cleaned.append({
                    "role": role,
                    "content": content,
                })

    return cleaned


# ==========================================
# TOKEN USAGE LOGGING
# ==========================================

def log_usage(response, agent_name):
    """Record token usage in Streamlit Cloud logs."""

    usage = getattr(response, "usage", None)

    if usage:

        print(
            f"[TOKEN USAGE] {agent_name}: "
            f"input={usage.input_tokens}, "
            f"output={usage.output_tokens}, "
            f"total={usage.total_tokens}"
        )


# ==========================================
# STANDARD GROQ LLM
# ==========================================

class GroqLLM(BaseLLM):

    def __init__(self, agent_name="Claim Analyst"):

        super().__init__(model=MODEL)

        self.client = create_client()
        self.agent_name = agent_name

    def call(
        self,
        messages,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs
    ):

        response = self.client.responses.create(
            model=MODEL,
            input=clean_messages(messages),
            max_output_tokens=700,
            reasoning={"effort": "low"},
        )

        log_usage(response, self.agent_name)

        result = response.output_text

        if not result:

            raise ValueError(
                f"{self.agent_name} returned an empty response."
            )

        return result


# ==========================================
# RESEARCH GROQ LLM
# ==========================================

class GroqResearchLLM(BaseLLM):

    def __init__(self):

        super().__init__(model=MODEL)

        self.client = create_client()

    def call(
        self,
        messages,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs
    ):

        # ----------------------------------
        # STEP 1: WEB RESEARCH
        # ----------------------------------

        response = self.client.responses.create(
            model=MODEL,
            input=clean_messages(messages),

            tools=[
                {"type": "browser_search"}
            ],

            tool_choice="required",
            max_tool_calls=1,

            max_output_tokens=1600,

            reasoning={"effort": "low"},
        )

        log_usage(response, "Research Agent")

        result = response.output_text or ""

        # ----------------------------------
        # STEP 2: EXTRACT SOURCE CITATIONS
        # ----------------------------------

        sources = []
        seen_urls = set()

        for item in response.output:

            if hasattr(item, "model_dump"):

                item_data = item.model_dump()

            elif isinstance(item, dict):

                item_data = item

            else:

                continue

            for content in item_data.get("content", []):

                if not isinstance(content, dict):
                    continue

                for annotation in content.get(
                    "annotations", []
                ):

                    if not isinstance(annotation, dict):
                        continue

                    if annotation.get("type") != "url_citation":
                        continue

                    url = annotation.get("url", "")
                    title = annotation.get("title", "")

                    if not url:
                        continue

                    if url in seen_urls:
                        continue

                    seen_urls.add(url)

                    sources.append({
                        "title": title or "Untitled source",
                        "url": url,
                    })

        # ----------------------------------
        # STEP 3: CHECK AND RETRIEVE SOURCES
        # ----------------------------------

        # HTTP requests only.
        # No additional Groq API calls.

        for source in sources[:MAX_RESEARCH_SOURCES]:

            source["accessibility"] = (
                "ACCESSIBILITY_UNCONFIRMED"
            )

            source["content_status"] = (
                "CONTENT_UNAVAILABLE"
            )

            source["excerpt"] = "NOT AVAILABLE"

            try:

                # Check basic URL accessibility.
                source["accessibility"] = check_source_url(
                    source["url"],
                    timeout=4,
                )

                # Retrieve readable webpage content.
                fetched = fetch_source_text(
                    source["url"],
                    timeout=5,
                    max_chars=12000,
                )

                source["content_status"] = fetched.get(
                    "status",
                    "CONTENT_UNAVAILABLE",
                )

                if (
                    source["content_status"]
                    == "CONTENT_RETRIEVED"
                ):

                    webpage_text = fetched.get(
                        "text", ""
                    )

                    if webpage_text:

                        source["excerpt"] = (
                            webpage_text[
                                :MAX_SOURCE_EXCERPT_CHARS
                            ]
                        )

            except Exception as error:

                print(
                    "[SOURCE RETRIEVAL ERROR] "
                    f"{type(error).__name__}"
                )

        # ----------------------------------
        # STEP 4: MARK UNCHECKED SOURCES
        # ----------------------------------

        for source in sources[MAX_RESEARCH_SOURCES:]:

            source["accessibility"] = (
                "NOT_CHECKED_LIMIT"
            )

            source["content_status"] = (
                "NOT_CHECKED_LIMIT"
            )

            source["excerpt"] = "NOT AVAILABLE"

        # ----------------------------------
        # STEP 5: APPEND SOURCE INFORMATION
        # ----------------------------------

        if sources:

            result += (
                "\n\n"
                "SOURCES RETURNED BY WEB SEARCH:\n"
            )

            for index, source in enumerate(
                sources,
                start=1,
            ):

                result += (
                    f"\nSOURCE {index}\n"
                    f"Title: {source['title']}\n"
                    f"URL: {source['url']}\n"
                    f"Accessibility: "
                    f"{source['accessibility']}\n"
                    f"Content retrieval: "
                    f"{source['content_status']}\n"
                    f"Retrieved webpage excerpt: "
                    f"{source['excerpt']}\n"
                )

            result += (
                "\nIMPORTANT EVIDENCE LIMITATIONS:\n"
                "- URL accessibility does not establish "
                "source credibility.\n"
                "- CONTENT_RETRIEVED means webpage text "
                "was obtained, not that a factual claim "
                "was verified.\n"
                "- Retrieved excerpts may contain "
                "navigation text or incomplete information.\n"
                "- Do not invent quotations or claim "
                "that a source supports a finding "
                "unless its relevant content establishes "
                "that relationship.\n"
                "- If evidence cannot be verified, "
                "clearly state the limitation.\n"
            )

        else:

            result += (
                "\n\nSOURCE VERIFICATION STATUS:\n"
                "No structured source citations were "
                "returned by the Groq API.\n"
                "Source-content verification "
                "was not performed.\n"
            )

        # ----------------------------------
        # STEP 6: LOG SOURCE RETRIEVAL
        # ----------------------------------

        checked_sources = sources[
            :MAX_RESEARCH_SOURCES
        ]

        retrieved_count = sum(
            1
            for source in checked_sources
            if source.get("content_status")
            == "CONTENT_RETRIEVED"
        )

        print(
            "[SOURCE VERIFICATION] "
            f"citations={len(sources)}, "
            f"checked={len(checked_sources)}, "
            f"content_retrieved={retrieved_count}"
        )

        # ----------------------------------
        # STEP 7: RETURN RESEARCH RESULT
        # ----------------------------------

        if not result.strip():

            raise ValueError(
                "Research Agent returned an empty response."
            )

        return result
