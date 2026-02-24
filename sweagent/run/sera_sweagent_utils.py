import os
import re

from jinja2 import Template
from openai import OpenAI, APIConnectionError, RateLimitError

def parse_trajectory(trajectory):
    """
    Parse trajectory file and information of each step.
    """

    formatted_steps = []
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        formatted_step = {
            "action": step.get("action", ""),
            "observation": step.get("observation", ""),
            "response": step.get("response", ""),
        }
        formatted_steps.append(formatted_step)
    return formatted_steps

def pp_regex(text, re_string=r"<output>(.*?)</output>"):
    matches = re.findall(re_string, text, re.DOTALL)
    if len(matches) == 0:
        return None
    return matches

def pp_query(system, prompt, model, base_url="", api_key="", max_tokens=4096, retries=0, args={}):
    # Create OpenAI-compatible client
    # Slice openai if it starts with
    if model.startswith("anthropic/"):
        model = model[len("anthropic/"):]
        base_url = "https://api.anthropic.com/v1/"
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if base_url != "":
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        max_tokens = max_tokens
    else:
        client = OpenAI()
        max_tokens = max_tokens
    if len(args) > 0:
        task_prompt = Template(prompt).render(**args)
        # print(task_prompt)
    else:
        task_prompt = prompt
    # print("Prompt:", task_prompt)
    # Make a request
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.6,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": task_prompt}
                ]
            )
            break
        except Exception as e:
            # print("=== ERROR ===")
            if retries == 0:
                raise
            time.sleep(30)
            retries -= 1
    return completion.choices[0].message.content