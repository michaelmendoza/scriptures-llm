# RAG System Improvements

Methods to improve response quality, organized by system component.

For detailed explanations of individual techniques, see [RAG.md](RAG.md).

---

## Ingestion Improvements

### Chunking Strategies

Experiment with chunk size, overlap, and chunking method. The way documents are split fundamentally determines what the retriever can find.

**Chunk size & overlap tuning**
- Larger chunks (1500–2000 chars) preserve more context but may dilute relevance for specific queries.
- Smaller chunks (300–500 chars) improve precision but lose surrounding context.
- Overlap (10–20% of chunk size) prevents information from being split across boundaries.
- There is no universal best size — tune based on your document type and query patterns.

**Context-aware chunking**
- **Sentence-aware**: Never break mid-sentence. Minimal improvement, trivial to implement.
- **Paragraph-aware**: Split on `\n\n` boundaries. Groups full paragraphs up to chunk size.
- **Section-aware (Markdown/HTML)**: Use headings (`##`, `<h2>`) as primary split points. Preserves topical coherence and keeps section titles attached to content.
- **Semantic chunking**: Use an embedding model to detect topic shifts between consecutive sentences. A sharp drop in embedding similarity indicates a natural split point. Most accurate, but significantly increases ingestion time.

### Contextual Retrieval

Introduced by Anthropic. Addresses the problem that chunks lose context when separated from their parent document.

- During ingestion, send each chunk + its parent document to an LLM.
- Prompt the LLM to generate a 2–3 sentence contextual prefix that situates the chunk (e.g., which document, section, topic it belongs to).
- Prepend the prefix to the chunk before embedding and storing.
- At query time, the enriched context improves both retrieval relevance and LLM answer quality.
- Anthropic reported up to 49% reduction in retrieval failure rate when combined with BM25.
- Trade-off: one LLM call per chunk during ingestion increases time and cost.

### Metadata Enrichment

Attach structured metadata to each chunk during ingestion for filtering at query time.

- **Source metadata**: filename, document title, section heading, URL, author, date.
- **Category tags**: topic, department, document type (e.g., "policy", "tutorial", "reference").
- **Structural metadata**: chunk position in document, parent section, heading hierarchy.
- Richer metadata enables filtered retrieval (e.g., only search within a specific category or date range), reducing noise and improving precision.

### Embedding Model Selection

The embedding model determines the quality of the vector representations.

- Larger embedding models (e.g., `mxbai-embed-large`, `snowflake-arctic-embed`) generally produce better representations than smaller ones.
- Domain-specific fine-tuned embeddings capture specialized terminology more accurately.
- Evaluate embedding quality by measuring retrieval recall on a test set — a better embedding model often matters more than retrieval tricks.

---

## Retrieval Improvements

### Hybrid Search

Combine dense vector search (embeddings) with sparse keyword search (BM25) for better recall.

- **Dense (vector) search**: Finds semantically similar content. Good at paraphrasing and conceptual matching. Can miss exact terminology.
- **Sparse (BM25) search**: Matches exact keywords and terms. Good at precise terminology. Misses paraphrased or conceptually similar content.
- Running both in parallel and merging results captures what either method alone would miss.
- Merge strategies: Reciprocal Rank Fusion (see below), weighted score combination, or interleaving.

### Reranking

After initial retrieval, rescore the candidate chunks before passing them to the LLM. Retrieve broadly (top 20–50), then rerank to a precise top-k.

**Reciprocal Rank Fusion (RRF)**
- Combines ranked lists from multiple retrieval methods (e.g., vector + BM25) into a single list.
- Scores each document based on its rank position across all lists: `RRF(doc) = Σ 1/(k + rank_i(doc))`.
- Documents appearing in multiple lists get boosted. No need to normalize scores across methods.
- Simple to implement, no additional model required.

**Cross-Encoder Re-Ranker**
- A model that takes a (query, document) pair and outputs a relevance score.
- Unlike bi-encoders (used for retrieval), cross-encoders process query and document together through full transformer attention, catching subtle relevance signals.
- Much more accurate than vector similarity, but too slow for initial search (~50ms per pair vs ~1ms).
- Use as a second stage: retrieve top-20 with vector search, rerank with cross-encoder, keep top-5.
- Consistently one of the highest-impact improvements to RAG quality.

### Metadata Filters

Use metadata attached during ingestion to narrow the search space at query time.

- Filter by source, category, date range, or document type before (or during) similarity search.
- Reduces noise by excluding irrelevant documents from the candidate pool.
- Can be applied automatically (based on query analysis) or manually (user-specified filters).
- ChromaDB supports `where` filters on metadata fields natively.

### Retrieval Parameter Tuning

- **Top-k**: Increasing from 5 to 8–10 provides more context, especially for questions spanning multiple sections. Too many chunks dilute relevance and waste LLM context.
- **Similarity threshold**: Filter out low-scoring chunks rather than always returning exactly k results. Avoids injecting irrelevant context.
- **Maximum Marginal Relevance (MMR)**: Select chunks that are both relevant to the query and diverse from each other, reducing redundant context.

---

## Query Improvements

### Query Reformulation

Use the LLM to rewrite the user's query into a more effective search query before retrieval.

- User queries are often conversational, vague, or use different vocabulary than the documents.
- A reformulated query aligned with document terminology retrieves more relevant chunks.
- Simple prompt: "Rewrite this question as a clear, specific search query: {question}"

### Query Expansion (Multi-Query)

Generate multiple query variants and retrieve results for all of them.

- Prompt the LLM to rephrase the question 2–3 different ways.
- Run retrieval for each variant and merge the results (using RRF or deduplication).
- Improves recall for ambiguously worded questions by covering different phrasings.

### HyDE (Hypothetical Document Embeddings)

Ask the LLM to generate a hypothetical answer, then embed *that* for retrieval instead of the original question.

- The hypothetical answer is closer in embedding space to actual document chunks than a short question.
- Especially useful when questions are very different in form from the documents they target.
- Trade-off: adds one LLM call before retrieval (latency), and a poor hypothetical answer can hurt results.

### Conversation Memory

For multi-turn chat, use conversation history to reformulate follow-up queries.

- "What about the second one?" is meaningless without prior context.
- Pass recent conversation history to the LLM and ask it to produce a standalone query.
- Enables natural conversational interactions over the knowledge base.

---

## Generation Improvements

### System Prompt Engineering

The system prompt shapes how the LLM uses the retrieved context.

- Emphasize coherence — "Synthesize information from multiple documents into a unified answer" rather than treating each chunk independently.
- Instruct the model to synthesize across multiple chunks into a unified answer, not just repeat the first relevant chunk.
- Add "think step by step" or chain-of-thought instructions for questions requiring reasoning.
- Include confidence instructions: ask the model to note when context is partial or uncertain.
- Specify citation format and enforce grounding: "Answer only from the provided context."
- Add a confidence signal — ask the model to note when context is partial or uncertain.

### LLM Model Selection

The choice of generation model directly impacts answer quality.

- Larger models (8B+) reason better, follow instructions more reliably, and produce more coherent answers.
- Evaluate models on your specific question set — the best model depends on your query complexity and domain.
- Consider the context window size: larger windows allow more retrieved chunks without truncation.

### Context Presentation

How retrieved chunks are formatted in the prompt affects the LLM's ability to use them.

- **Ordering**: Place the most relevant chunks first (or last, depending on model — some models attend more to the beginning or end of the context).
- **Source labels**: Clearly label each chunk with its source so the LLM can cite accurately.
- **Deduplication**: Remove near-duplicate chunks before injection to avoid wasting context window.
- **Summarization**: For very long contexts, summarize retrieved chunks before injection to fit within limits.

### Prompt Chaining

Instead of a single LLM call, break generation into multiple sequential steps where each step's output feeds the next. Decomposes a complex task into focused subtasks, improving accuracy at each stage.

**Query Reformulation → Retrieve → Generate**
- Step 1: LLM rewrites the user's conversational question into an optimized search query aligned with document terminology.
- Step 2: Retrieve using the reformulated query.
- Step 3: LLM generates the final answer from retrieved chunks.
- Improves retrieval quality by bridging the vocabulary gap between user language and document language.

**Retrieve → Analyze → Generate**
- Step 1: Retrieve chunks normally.
- Step 2: LLM extracts and summarizes the key facts relevant to the question from the raw chunks, with source attributions.
- Step 3: LLM generates a coherent final answer from the structured facts (not the raw chunks).
- Separates "understanding the context" from "writing the answer", letting each step focus on one task.

**Full chain (Reformulate → Retrieve → Analyze → Generate)**
- Combines both chains for maximum quality.
- 3 LLM calls per query — trade-off is increased latency.

**When it helps**
- Questions that use different vocabulary than the source documents (reformulation).
- Large or noisy retrieved contexts where the LLM struggles to find the relevant parts (analyze step).
- Complex questions requiring synthesis across multiple chunks.

**Trade-offs**
- Each additional LLM call adds latency (significant with local models).
- Errors can compound — a bad reformulation leads to bad retrieval.
- Best paired with a config toggle so it can be disabled for simple queries.

### Parent-Child Chunking

Retrieve small chunks for precision, but pass the larger parent chunk to the LLM for context.

- Store two levels: small child chunks (200–300 chars) for retrieval, and their larger parent chunks (1000+ chars) for generation.
- When a child chunk matches, look up and pass its parent to the LLM.
- The LLM gets surrounding context that helps it produce more complete answers.

---

## Evaluation & Monitoring

### Test Set

Build a set of 20–50 question-answer pairs from your documents and measure answer quality over time.

- Tracks whether changes actually improve the system or introduce regressions.
- Include a mix of easy, hard, and multi-hop questions.
- Update the test set as you add new documents.

### Retrieval Metrics

Measure whether the correct source document appears in retrieved results.

- **Recall@k**: Does the relevant chunk appear in the top-k?
- **Precision@k**: What fraction of retrieved chunks are actually relevant?
- **MRR (Mean Reciprocal Rank)**: How high does the first relevant chunk rank?

### LLM-as-Judge

Use a second LLM call to evaluate generated answers.

- Check faithfulness: is the answer supported by the retrieved context, or does it hallucinate?
- Check completeness: does the answer address the full question?
- Check relevance: is the answer focused or does it include unnecessary information?

### RAGAS Framework

Automated RAG evaluation measuring multiple quality dimensions.

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer relevance**: Does the answer address the question?
- **Context precision**: Are the retrieved chunks actually relevant?
- **Context recall**: Were all necessary chunks retrieved?
