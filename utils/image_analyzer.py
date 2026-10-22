import base64
import streamlit as st
from openai import OpenAI


VISION_MODEL = "qwen/qwen3.8-27b"


def analyze_image(image_file, related_claim=""):
    """
    Analyze an uploaded image using Groq's vision model.

    This describes and assesses visible evidence.
    It does not prove whether an image is authentic.
    """

    # Read the uploaded image
    image_bytes = image_file.getvalue()

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    # Prevent unnecessarily large API requests
    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError(
            "Please upload an image smaller than 10 MB."
        )

    # Accept only supported image formats
    image_type = image_file.type

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]

    if image_type not in allowed_types:
        raise ValueError(
            "Please upload a JPG, PNG, or WebP image."
        )

    # Convert the image into a format accepted by the API
    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:{image_type};base64,{encoded_image}"
    )

    # Create the Groq client using your existing secret
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    # Give the model a focused media-verification task
    prompt = """
    You are the Image Analysis component of TruthLens AI.

    Analyze the supplied image carefully and produce a
    concise, evidence-based report with these sections:

    1. VISIBLE CONTENT
       Describe the people, objects, scene, and other
       details that are actually visible.

    2. VISIBLE TEXT
       Transcribe important readable text, if any.
       Clearly identify text that is unclear.

    3. CONTEXT AND CLAIM
       Explain what the image appears to show and whether
       the image itself provides evidence relevant to
       the user's claim.

    4. POSSIBLE WARNING SIGNS
       Mention visible inconsistencies or suspicious
       details only when you can identify specific
       observations. Do not treat unusual appearance
       alone as proof of manipulation.

    5. LIMITATIONS
       State what cannot be established from the image.
       Do not claim that the image is AI-generated,
       edited, authentic, or fake without reliable evidence.

    6. PRELIMINARY ASSESSMENT
       Use one of these labels:
       - Relevant visual evidence found
       - Possible warning signs observed
       - Insufficient visual evidence

    IMPORTANT:
    - Do not invent details that are not visible.
    - A photograph alone may not establish when or where
      it was taken.
    - Do not identify an unknown person by guessing.
    - Do not make a definitive authenticity judgment.
    - Separate observations from assumptions.
    """

    if related_claim.strip():
        prompt += (
            "\n\nUSER'S RELATED CLAIM:\n"
            + related_claim.strip()
            + "\n\nAssess whether the visible image supports "
              "that claim. Do not assume the claim is true."
        )

    # Send the image and instructions to Groq
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
        max_completion_tokens=900,
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "The image model returned an empty response."
        )

    return result
