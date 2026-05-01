import json
import os
import re

import ollama

MODEL_NAME = "llama3"
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
DEFAULT_REASONING_PATHS = 6
GENERATION_PROFILES = {
    "fast": {
        "options": {
            "temperature": 0.2,
            "top_p": 0.8,
            "num_ctx": 768,
            "num_predict": 120,
        },
        "word_limit": 16,
        "extra_rule": "Keep each path short and direct.",
        "multiline": False,
    },
    "detailed": {
        "options": {
            "temperature": 0.35,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": 700,
        },
        "word_limit": 140,
        "extra_rule": "Give 5 to 6 short lines of step-by-step explanation before the final answer.",
        "multiline": True,
    },
}


def _extract_json_block(content):
    start = content.find("[")
    end = content.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model response.")

    return content[start:end + 1]


def _extract_marked_paths(content, k):
    pattern = re.compile(
        r"Path\s*\d+\s*:\s*(.*?)(?=Path\s*\d+\s*:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    paths = [match.strip(" -\n\r\t") for match in pattern.findall(content)]
    return [path for path in paths if path][:k]


def _extract_block_paths(content, k):
    pattern = re.compile(
        r"Path\s*\d+\s*:\s*(.*?)(?=Path\s*\d+\s*:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    paths = [match.strip() for match in pattern.findall(content)]
    return [path for path in paths if path][:k]


def _path_template(k, multiline):
    separator = "\n\n" if multiline else "\n"
    return separator.join(f"Path {index}: ..." for index in range(1, k + 1))


def generate_reasoning(question, k=DEFAULT_REASONING_PATHS, mode="fast"):
    profile = GENERATION_PROFILES.get(mode, GENERATION_PROFILES["fast"])
    path_template = _path_template(k, profile["multiline"])

    if profile["multiline"]:
        prompt = f"""Return exactly {k} reasoning blocks and nothing else.
Use exactly this format:
{path_template}

Rules:
- Each path must be under {profile["word_limit"]} words.
- Include the final answer in each path.
- {profile["extra_rule"]}
- No intro sentence.
- No markdown.
- Leave a blank line between paths.

Question: {question}
"""
    else:
        prompt = f"""Return exactly {k} lines and nothing else.
Use exactly this format:
{path_template}

Rules:
- Each path must be under {profile["word_limit"]} words.
- Include the final answer in each path.
- {profile["extra_rule"]}
- No intro sentence.
- No markdown.

Question: {question}
"""

    client = (
    ollama.Client(
        host=OLLAMA_HOST,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    if OLLAMA_HOST
    else ollama.Client()
)

    response = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options=profile["options"],
        keep_alive="15m",
    )

    content = response["response"].strip()

    try:
        items = json.loads(_extract_json_block(content))
        paths = [item["reasoning"].strip() for item in items if item.get("reasoning")]
    except (ValueError, KeyError, json.JSONDecodeError):
        if profile["multiline"]:
            paths = _extract_block_paths(content, k)
        else:
            paths = _extract_marked_paths(content, k)

    if not paths:
        paths = [part.strip() for part in content.split("\n\n") if part.strip()]

    return paths[:k]


def get_model_label():
    if OLLAMA_HOST:
        return f"{MODEL_NAME} via Ollama ({OLLAMA_HOST})"
    return f"{MODEL_NAME} via local Ollama"
