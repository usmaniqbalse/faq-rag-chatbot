import ollama
from prompts.system_prompt import SYSTEM_PROMPT


def call_llm(context: str, prompt: str):
    """
    Generates a response from the language model based on context and prompt.

    Args:
        context: Contextual information for the question.
        prompt: User's question.

    Yields:
        Chunks of the generated response.
    """
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break
