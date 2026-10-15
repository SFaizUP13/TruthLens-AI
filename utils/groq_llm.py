import streamlit as st
from openai import OpenAI
from crewai.llms.base_llm import BaseLLM


MODEL = "openai/gpt-oss-20b"


def create_client():
    return OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )


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
                    "content": content
                })

    return cleaned


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
            reasoning={"effort": "low"}
        )

        log_usage(response, self.agent_name)

        result = response.output_text

        if not result:
            raise ValueError(
                "Claim Analyst returned an empty response."
            )

        return result


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
        response = self.client.responses.create(
            model=MODEL,
            input=clean_messages(messages),

            # Enable current web research.
            tools=[
                {"type": "browser_search"}
            ],

            # Research must use the search tool.
            tool_choice="required",

            # Limit research calls per response.
            max_tool_calls=1,

            # Limit generated and reasoning tokens.
            max_output_tokens=1600,
            reasoning={"effort": "low"}
        )

    log_usage(response, "Research Agent")

    result = response.output_text or ""

# Collect source citations returned by the API.
sources = []

for item in response.output:
    # Convert SDK objects to dictionaries where supported.
    if hasattr(item, "model_dump"):
        item_data = item.model_dump()
    elif isinstance(item, dict):
        item_data = item
    else:
        continue

    for content in item_data.get("content", []):
        if not isinstance(content, dict):
            continue

        for annotation in content.get("annotations", []):
            if not isinstance(annotation, dict):
                continue

            if annotation.get("type") == "url_citation":
                url = annotation.get("url", "")
                title = annotation.get("title", "")

                if url and url not in [
                    source["url"] for source in sources
                ]:
                    sources.append({
                        "title": title or "Untitled source",
                        "url": url
                    })

        # Append citations only when the API actually returns them.
        if sources:
            result += "\n\nSOURCES RETURNED BY WEB SEARCH:\n"

            for index, source in enumerate(sources, start=1):
                result += (
                    f"\n{index}. {source['title']}"
                    f"\n   URL: {source['url']}\n"
                )

        if not result.strip():
            raise ValueError("Research Agent returned an empty response.")

        return result
