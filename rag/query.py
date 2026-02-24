"""Query reformulation: rewrite conversational queries for better retrieval."""

import ollama

from rag import config


def reformulate_query(user_query: str) -> str:
    """Rewrite a conversational query into an optimized search query.

    Uses the LLM to strip conversational phrasing and focus the query on the
    key information need, producing a query better suited for semantic search.
    """
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. Rewrite the user's conversational "
                    "question into a concise, keyword-rich search query optimized for "
                    "semantic search over a document collection. "
                    "Rules:\n"
                    "- Remove conversational filler (e.g. 'Can you tell me', 'I want to know')\n"
                    "- Keep all important entities, names, and concepts\n"
                    "- Output ONLY the reformulated query, nothing else\n"
                    "- Do not add quotes or explanation"
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )
    return response["message"]["content"].strip()
