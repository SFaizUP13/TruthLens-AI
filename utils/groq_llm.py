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
            max_output_tokens=700,
            reasoning={"effort": "low"}
        )

        log_usage(response, "Claim Analyst")

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

        result = response.output_text

        if not result:
            raise ValueError(
                "Research Agent returned an empty response."
            )

        return result
