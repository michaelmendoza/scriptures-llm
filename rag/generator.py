import ollama

from rag import config

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Rules:
- Answer based ONLY on the provided context below.
- Cite your sources using [source: filename] format after relevant statements.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer that question."
- Be concise and direct."""


def build_prompt(query: str, contexts: list[dict]) -> list[dict]:
    """Build the message list for the Ollama chat API."""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n--- Document {i} [source: {ctx['source']}] ---\n"
        context_text += ctx["text"]
        context_text += "\n"

    user_message = f"""Context:
{context_text}

Question: {query}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate(query: str, contexts: list[dict]) -> str:
    """Generate a response using Ollama, streaming to stdout."""
    messages = build_prompt(query, contexts)

    full_response = ""
    for chunk in ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token

    print()  # newline after streaming
    return full_response
