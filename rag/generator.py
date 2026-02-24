"""LLM generation: prompt building, enhanced prompts, context presentation, and prompt chaining."""

import ollama

from rag import config

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Rules:
- Answer based ONLY on the provided context below.
- Cite your sources using [source: filename] format after relevant statements.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer that question."
- Be concise and direct."""

ENHANCED_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context documents.

Rules:
- Answer based ONLY on the provided context below.
- Think step-by-step: first identify the relevant pieces of information, then synthesize your answer.
- When multiple documents discuss the same topic, synthesize information across them rather than repeating each one.
- Cite your sources using [source: filename] format after relevant statements.
- If documents conflict, acknowledge the discrepancy and present both perspectives.
- Express confidence levels: use phrases like "clearly stated", "implied by", or "not directly addressed" as appropriate.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer that question."
- Be thorough but concise."""


# ---------------------------------------------------------------------------
# Context presentation helpers
# ---------------------------------------------------------------------------

def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _prepare_contexts(contexts: list[dict]) -> list[dict]:
    """Deduplicate and order contexts for presentation.

    - Remove chunks with >90% Jaccard word similarity
    - Sort by best available score (rerank > rrf > distance)
    """
    if not contexts:
        return []

    # Deduplicate
    unique: list[dict] = []
    for ctx in contexts:
        is_dup = False
        for existing in unique:
            if _jaccard_similarity(ctx["text"], existing["text"]) > 0.9:
                is_dup = True
                break
        if not is_dup:
            unique.append(ctx)

    # Sort by best score — handle both similarity (higher=better) and distance (lower=better)
    def sort_key(c):
        for key in ("rerank_score", "rrf_score"):
            val = c.get(key) or c.get("metadata", {}).get(key)
            if val is not None:
                return -val  # Higher is better → negate for ascending sort
        # Default distance score (lower is better)
        score = c.get("score")
        return score if score is not None else float("inf")

    unique.sort(key=sort_key)
    return unique


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(query: str, contexts: list[dict]) -> list[dict]:
    """Build the message list for the Ollama chat API."""
    # Select system prompt
    system_prompt = ENHANCED_SYSTEM_PROMPT if config.ENABLE_ENHANCED_PROMPT else SYSTEM_PROMPT

    # Optionally prepare contexts (deduplicate + reorder)
    if config.ENABLE_CONTEXT_PRESENTATION:
        contexts = _prepare_contexts(contexts)

    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        # Build document label
        label = f"source: {ctx['source']}"
        meta = ctx.get("metadata", {})
        if config.ENABLE_CONTEXT_PRESENTATION and config.ENABLE_METADATA_ENRICHMENT:
            parts = []
            if meta.get("document_title"):
                parts.append(meta["document_title"])
            if meta.get("section_heading"):
                parts.append(meta["section_heading"])
            if parts:
                label = " > ".join(parts) + f" [{ctx['source']}]"

        context_text += f"\n--- Document {i} [{label}] ---\n"
        context_text += ctx["text"]
        context_text += "\n"

    user_message = f"""Context:
{context_text}

Question: {query}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt chaining: analyze → extract → generate
# ---------------------------------------------------------------------------

def analyze_and_extract(query: str, contexts: list[dict]) -> str:
    """Step 1 of prompt chaining: extract key facts from retrieved chunks.

    Returns a structured list of facts with source attributions.
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n--- Document {i} [source: {ctx['source']}] ---\n"
        context_text += ctx["text"]
        context_text += "\n"

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact extractor. Given a question and source documents, "
                    "extract all facts relevant to answering the question. "
                    "Format each fact as:\n"
                    "- FACT: <the fact> [source: <filename>]\n\n"
                    "Only extract facts that are directly relevant to the question. "
                    "Do not make inferences or add information not in the documents."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocuments:\n{context_text}",
            },
        ],
    )
    return response["message"]["content"].strip()


def generate_from_facts(query: str, facts: str) -> str:
    """Step 2 of prompt chaining: generate final answer from extracted facts.

    Streams the response to stdout.
    """
    full_response = ""
    for chunk in ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Generate a comprehensive answer to the "
                    "question using ONLY the extracted facts below. "
                    "Cite sources using [source: filename] format. "
                    "Be concise and well-organized."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nExtracted Facts:\n{facts}",
            },
        ],
        stream=True,
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token

    print()  # newline after streaming
    return full_response
