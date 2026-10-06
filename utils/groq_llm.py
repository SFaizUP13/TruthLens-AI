import streamlit as st
from openai import OpenAI
from crewai.llms.base_llm import BaseLLM


class GroqLLM(BaseLLM):

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
            input=messages
        )

        return response.output_text

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

        # Clean CrewAI messages before sending them to Groq
        clean_messages = []

        for message in messages:

            clean_message = {
                "role": message.get("role"),
                "content": message.get("content", "")
            }

            clean_messages.append(clean_message)

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=clean_messages,
            tools=[
                {
                    "type": "browser_search"
                }
            ],
            tool_choice="required"
        )

        result = response.choices[0].message.content

        if not result:
            raise ValueError(
                "Groq Research Agent returned an empty response."
            )

        return result
