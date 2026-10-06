import streamlit as st

from openai import OpenAI
from crewai.llms.base_llm import BaseLLM


class GroqLLM(BaseLLM):

    def __init__(self):

        super().__init__(
            model="openai/gpt-oss-20b"
        )

        self.client = OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1"
        )

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

            model="openai/gpt-oss-20b",

            input=messages
        )
class GroqResearchLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="openai/gpt-oss-20b")

        self.client = OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1"
        )

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
            model="openai/gpt-oss-20b",
            input=messages,
            tool_choice="required",
            tools=[
                {
                    "type": "browser_search"
                }
            ]
        )

        return response.output_text

        return response.output_text


