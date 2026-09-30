from openai import OpenAI
from paperlens.config import settings

client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=120)

def generate(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0,
    )
    return response.choices[0].message.content or "No response generated."
