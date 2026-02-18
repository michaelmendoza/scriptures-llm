# RAG Improvement Techniques — Detailed Reference

A deep-dive reference for techniques that improve upon vanilla RAG. Each section explains how the technique works, why it matters, and how to implement it.

## Table of Contents
- [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
- [Cross-Encoder Re-Ranker](#cross-encoder-re-ranker)
- [Contextual Retrieval](#contextual-retrieval)
- [Context-Aware Chunking](#context-aware-chunking)
- [Knowledge Graphs](#knowledge-graphs)
- [Other Techniques](#other-techniques)
- [Evaluation & Quality](#evaluation--quality)

---

## Reciprocal Rank Fusion (RRF)

RRF is a technique for combining results from multiple retrieval methods (e.g., semantic search + BM25 keyword search) into a single, superior ranked list.

### How it works
Each retrieval method returns its own ranked list. RRF scores each document based on its rank position across all lists using the formula:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
```

Where `k` is a constant (typically 60) and `rank_i` is the document's position in retrieval method `i`. Documents appearing in multiple lists get their scores summed, naturally boosting items that multiple methods agree are relevant.

### Why it matters
- Semantic search finds conceptually similar content but can miss exact terminology
- Keyword search (BM25) catches precise term matches but misses paraphrased content
- RRF combines both without needing to normalize scores across different methods
- Simple to implement: just sort by combined RRF score after running both retrievers

### Implementation approach
1. Run vector search (ChromaDB) → get top-20 results with ranks
2. Run BM25 keyword search (`rank-bm25` library) → get top-20 results with ranks
3. Compute RRF score for each unique document across both lists
4. Sort by RRF score, take top-k

---

## Cross-Encoder Re-Ranker

A cross-encoder is a model that takes a (query, document) pair as input and outputs a relevance score. Unlike bi-encoders (used for initial retrieval), cross-encoders see both query and document simultaneously, enabling much deeper understanding of relevance.

### How it works
```
Initial Retrieval (fast, approximate)     Re-Ranking (slow, precise)
        |                                          |
Bi-encoder embeds query      ->  Top 20  ->  Cross-encoder scores each
and docs separately                           (query, chunk) pair
        |                                          |
   ~1ms per doc                               ~50ms per pair
                                                   |
                                              Top 5 (high quality)
```

### Why it matters
- Bi-encoders compress query and document into fixed-size vectors independently — this loses nuance
- Cross-encoders process query and document together through full transformer attention, catching subtle relevance signals (negation, conditional statements, specific relationships)
- A cross-encoder on top-20 candidates catches mistakes the initial retrieval makes
- This is consistently one of the highest-impact improvements to RAG quality

### Implementation approach
1. Retrieve top-20 candidates from ChromaDB (fast bi-encoder search)
2. Score each candidate with a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers, or a local model via Ollama)
3. Re-sort by cross-encoder score, take top-5
4. Pass only the re-ranked top-5 to the LLM

---

## Contextual Retrieval

Contextual retrieval (introduced by Anthropic) addresses a fundamental problem: chunks lose context when separated from their parent document. A chunk saying "the company increased revenue by 15%" is meaningless without knowing *which* company and *which year*.

### How it works
Before embedding each chunk, use an LLM to generate a short contextual prefix that situates the chunk within its source document:

```
Original chunk:
  "Revenue increased by 15% compared to the previous quarter."

Contextualized chunk:
  "This chunk is from Acme Corp's Q3 2024 earnings report, discussing
   financial performance. Revenue increased by 15% compared to the
   previous quarter."
```

The contextual prefix is prepended to the chunk before embedding and storage. At query time, the enriched chunks are retrieved and the full contextualized text is passed to the LLM.

### Why it matters
- Chunks often contain pronouns, relative references, and implicit context that only makes sense in the full document
- Embedding the contextualized chunk produces much better vectors for retrieval
- The LLM receives richer context, reducing hallucination and improving answer quality
- Anthropic reported up to 49% reduction in retrieval failure rate when combined with BM25

### Implementation approach
1. During ingestion, for each chunk, send the full document (or a large window) + the chunk to an LLM
2. Prompt: "Give a short context (2-3 sentences) situating this chunk within the document"
3. Prepend the generated context to the chunk
4. Embed and store the contextualized chunk
5. Trade-off: increases ingestion time and cost (one LLM call per chunk) but significantly improves retrieval quality

---

## Context-Aware Chunking

Standard chunking splits text at fixed character/token boundaries, often breaking mid-sentence or mid-paragraph. Context-aware chunking respects the natural structure of documents.

### How it works
Instead of splitting every N characters, analyze document structure and split at meaningful boundaries:

```
Fixed Chunking (naive):              Context-Aware Chunking:
+------------------+                 +------------------+
| ...end of sect.  |                 | ## Section 1     |
| ## Section 2     |  <- split mid-  | Full paragraph.  |
| First paragraph  |    section      | Another para.    |
| of section 2...  |                 +------------------+
+------------------+                 +------------------+
                                     | ## Section 2     |
                                     | First paragraph  |
                                     | of section 2...  |
                                     +------------------+
```

### Strategies (from simple to advanced)

1. **Sentence-aware**: Split at sentence boundaries (period + space), never mid-sentence. Minimal improvement but trivial to implement.

2. **Paragraph-aware**: Split at paragraph boundaries (`\n\n`). Group consecutive paragraphs up to the chunk size limit. Good balance of simplicity and quality.

3. **Section-aware (Markdown/HTML)**: Use headings (`##`, `<h2>`) as primary split points. Each section becomes a chunk (or multiple chunks if the section is long). Preserves topical coherence.

4. **Semantic chunking**: Use an embedding model to detect where the topic shifts. Compute embedding similarity between consecutive sentences — a sharp drop in similarity indicates a good split point. Most accurate but slowest.

### Why it matters
- Chunks that contain complete thoughts retrieve better than fragments
- A chunk that starts mid-sentence or mixes two topics produces a poor embedding
- Markdown files naturally have structure (headers, lists) — use it
- Preserving section headers in chunks gives the LLM important context about what the chunk is about

### Implementation approach
1. For Markdown: parse headers and split at `## ` boundaries first
2. Within each section, split at paragraph boundaries (`\n\n`) if section exceeds chunk size
3. Within each paragraph group, split at sentence boundaries as a last resort
4. Always include the section header at the top of each chunk for context
5. Apply overlap at paragraph boundaries, not mid-sentence

---

## Knowledge Graphs

Knowledge graphs represent information as entities (nodes) and relationships (edges), enabling structured reasoning that vector search alone cannot provide.

### How it works
```
Document: "Dr. Smith leads the cardiology department at City Hospital.
           She published a study on heart failure treatments in 2024."

Extracted Knowledge Graph:
  (Dr. Smith) --leads--> (Cardiology Department)
  (Cardiology Department) --part_of--> (City Hospital)
  (Dr. Smith) --published--> (Heart Failure Study)
  (Heart Failure Study) --year--> (2024)
  (Heart Failure Study) --topic--> (Heart Failure Treatments)
```

At query time, the system can traverse the graph to answer multi-hop questions that vector search struggles with:
- "Who leads the department that heart failure research comes from?" → Traverse: Heart Failure Study → Cardiology Department → Dr. Smith

### Why it matters
- Vector search finds *similar* content but can't reason about *relationships*
- Multi-hop questions ("What department does the person who published X belong to?") require connecting information across multiple chunks
- Knowledge graphs provide structured, traversable relationships
- Combining graph traversal with vector search (GraphRAG) gives both semantic similarity and relational reasoning

### Implementation approaches

1. **LLM-based extraction**: For each chunk, prompt an LLM to extract (subject, predicate, object) triples. Store in a graph database (Neo4j) or simple in-memory graph (NetworkX).

2. **Hybrid retrieval**: At query time, extract entities from the question, look them up in the graph, traverse 1-2 hops for related entities, then use those entities to filter or augment vector search results.

3. **GraphRAG (Microsoft)**: Builds a hierarchical community structure from the knowledge graph. Summarizes communities at different levels. For broad questions, uses community summaries; for specific questions, uses entity-level graph traversal + vector search.

### Trade-offs
- Significantly increases ingestion complexity and time
- Requires entity resolution (is "Dr. Smith" the same as "Jane Smith"?)
- Graph quality depends heavily on LLM extraction accuracy
- Best suited for corpora with rich entity relationships (medical, legal, organizational docs)
- Overkill for simple document Q&A — most valuable when questions require connecting information across documents

---

## Other Techniques

- **HyDE (Hypothetical Document Embeddings)**: Ask the LLM to generate a hypothetical answer, then embed *that* for retrieval. The hypothetical answer is closer in embedding space to actual document chunks than the original question.
- **Query Expansion**: Rephrase the user query 2-3 ways and search for all variations. Improves recall for ambiguously worded questions.
- **Parent-Child Chunking**: Store small chunks (200 chars) for precise retrieval, but pass the larger parent chunk (1000+ chars) to the LLM for richer context.
- **Conversation Memory**: Maintain chat history and reformulate queries using prior context so "What about the second one?" becomes meaningful.
- **Agentic RAG**: Let the LLM decide *when* and *what* to retrieve dynamically, rather than always following a fixed pipeline.
- **Fine-tuned Embeddings**: Train the embedding model on your domain data for better capture of specialized terminology.

---

## Evaluation & Quality
- **Build a test set**: Create 20-50 question-answer pairs from your documents. Run them through the system and measure answer quality over time.
- **Retrieval metrics**: Track whether the correct source document appears in top-k results (recall@k, precision@k, MRR).
- **LLM-as-judge**: Use a second LLM call to evaluate whether the generated answer is faithful to the retrieved context (detects hallucination).
- **RAGAS framework**: Automated evaluation measuring faithfulness, answer relevance, and context precision.
