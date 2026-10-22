import base64
import re

import streamlit as st
from openai import OpenAI


VISION_MODEL = "qwen/qwen3.8-27b"


def analyze_image(image_file, related_claim=""):
    """
    Analyze an image and identify a factual claim
    that can be passed to the fact-checking pipeline.
    """

    image_bytes = image_file.getvalue()

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError(
            "Please upload an image smaller than 10 MB."
        )

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

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:{image_type};base64,{encoded_image}"
    )

    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = """
    You are the Image Analysis component of TruthLens AI.

    Analyze the supplied image and produce a concise report.

    1. VISIBLE CONTENT
    Describe the image and its visible elements.

    2. VISIBLE TEXT
    Transcribe important readable text.
    Mark unclear text as uncertain.

    3. CONTEXT AND CLAIM
    Explain the central message and factual claims
    presented by the image.

    4. POSSIBLE WARNING SIGNS
    Identify specific observable inconsistencies,
    missing context, or other relevant warning signs.
    Do not treat these observations as proof of fraud.

    5. LIMITATIONS
    Explain what cannot be established from the image.
    Do not claim the image is fake or AI-generated
    without reliable evidence.

    6. PRELIMINARY ASSESSMENT
    Choose one:
    - Relevant visual evidence found
    - Possible warning signs observed
    - Insufficient visual evidence

    7. MAIN CLAIM FOR FACT-CHECKING
    Extract ONE concise, independently verifiable
    factual claim from the image.

    Output the final section using EXACTLY this format,
    on a single line:

    MAIN CLAIM FOR FACT-CHECKING: [one factual sentence]

    If no clear factual claim can be extracted, write:

    MAIN CLAIM FOR FACT-CHECKING: No clear factual claim identified.

    IMPORTANT:
    - Preserve the meaning of the original claim.
    - Do not turn questions, opinions, or speculation
      into established facts.
    - Do not invent names, dates, or locations.
    - If the claim is ambiguous, say so.
    - Extracting a claim does not mean it is true.
    - Do not make a final truth verdict.
    """

    if related_claim.strip():
        prompt += (
            "\n\nUSER'S ADDITIONAL CONTEXT:\n"
            + related_claim.strip()
            + "\nUse this context to understand the image, "
              "but do not assume it is true."
        )

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
        max_completion_tokens=1000,
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "The image model returned an empty response."
        )

    return result


def extract_main_claim(image_report):
    """
    Extract the explicitly labelled claim from the report.
    Return an empty string if no claim is available.
    """

    import re

    pattern = (
        r"^\s*MAIN CLAIM FOR FACT-CHECKING:\s*(.+?)\s*$"
    )

    match = re.search(
        pattern,
        image_report,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not match:
        return ""

    claim = match.group(1).strip()

    if claim.lower() == "no clear factual claim identified.":
        return ""

    return claim
    if not match:
        return ""

    claim = match.group(1).strip()

    if claim.lower() == "no clear factual claim identified.":
        return ""

    return claim
