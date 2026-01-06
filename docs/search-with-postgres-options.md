# Search Options for Postgres-Only Deployment

This document outlines approaches for implementing substring search on inference inputs/outputs when large content is stored outside Postgres for write performance.

## Problem Statement

When storing large inference outputs (1-2MB) directly in Postgres:
- Write throughput is limited to ~60-80 RPS due to CPU overhead (TOAST compression, JSONB processing)
- At 2MB per request, Postgres CPU becomes the bottleneck

To achieve higher write throughput, we can store large content in object storage (S3) while keeping metadata in Postgres. However, this creates a challenge for substring search on the content.

## Options

### Option 1: Postgres + pg_trgm (Excerpt-Based Search)

Store a truncated/sampled excerpt in Postgres for search, full content in S3.

```sql
CREATE TABLE chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,

    -- Searchable excerpts (e.g., first 10KB)
    input_excerpt TEXT,
    output_excerpt TEXT,

    -- References to full content in S3
    input_s3_key TEXT,
    output_s3_key TEXT,

    -- Other metadata
    inference_params JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER,
    timestamp TIMESTAMPTZ NOT NULL
);

-- Trigram index for substring search (LIKE '%foo%')
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_input_trgm ON chat_inference USING GIN (input_excerpt gin_trgm_ops);
CREATE INDEX idx_output_trgm ON chat_inference USING GIN (output_excerpt gin_trgm_ops);

-- Example query
SELECT * FROM chat_inference
WHERE input_excerpt ILIKE '%search term%'
   OR output_excerpt ILIKE '%search term%';
```

**Pros:**
- Single database, simple architecture
- No additional services to manage
- Transactionally consistent

**Cons:**
- Can only search the excerpt, not full content
- Need to decide truncation strategy (first N bytes? key portions?)
- pg_trgm indexes can be large for text columns

**Best for:** Cases where most searches match content in the first portion of the text.

---

### Option 2: Elasticsearch/OpenSearch (Full-Text Search)

Dedicated search engine for full content indexing, Postgres for metadata.

```
Write Path:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Postgres   │     │     S3      │     │Elasticsearch│
│  (metadata) │     │  (blobs)    │     │  (search)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                    │
      ▼                   ▼                    ▼
  1. Write         2. Write full        3. Async index
     metadata         content              content

Search Path:
  1. Query Elasticsearch for matching inference IDs
  2. Fetch metadata from Postgres
  3. Fetch full content from S3 if needed
```

**Schema in Postgres:**
```sql
CREATE TABLE chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,

    -- S3 references only, no content
    input_s3_key TEXT NOT NULL,
    output_s3_key TEXT NOT NULL,

    -- Metadata
    inference_params JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER,
    timestamp TIMESTAMPTZ NOT NULL
);
```

**Elasticsearch Index:**
```json
{
  "mappings": {
    "properties": {
      "inference_id": { "type": "keyword" },
      "input": { "type": "text" },
      "output": { "type": "text" },
      "function_name": { "type": "keyword" },
      "timestamp": { "type": "date" }
    }
  }
}
```

**Pros:**
- Full substring search on entire content
- Purpose-built for search, scales independently
- Rich query capabilities (fuzzy matching, relevance scoring)

**Cons:**
- Another service to manage (operational complexity)
- Eventual consistency for search (async indexing)
- Additional infrastructure cost

**Best for:** Production deployments needing full-text search at scale.

---

### Option 3: ClickHouse for Search (Hybrid Architecture)

Keep ClickHouse specifically for search/analytics, use Postgres for transactional data.

```
┌─────────────────┐           ┌─────────────────┐
│    Postgres     │           │   ClickHouse    │
│  (OLTP writes)  │──────────▶│ (search/OLAP)   │
│  - metadata     │   async   │ - full content  │
│  - rate limits  │   sync    │ - analytics     │
│  - experiments  │           │ - search        │
└─────────────────┘           └─────────────────┘
         │
         ▼
   Critical path
   (low latency)
```

ClickHouse has excellent substring search capabilities:
```sql
-- Fast substring search in ClickHouse
SELECT inference_id, function_name, timestamp
FROM chat_inferences
WHERE position(output, 'search term') > 0
   OR hasSubsequence(input, 'search term');

-- With LIKE
SELECT * FROM chat_inferences
WHERE output LIKE '%error%';
```

**Pros:**
- Existing infrastructure (if already using ClickHouse)
- Excellent performance for large text search and analytics
- Can store full content efficiently (compression)
- Combines search + analytics in one system

**Cons:**
- Still requires ClickHouse dependency
- Async replication means eventual consistency for search
- Two databases to manage

**Best for:** Deployments already using ClickHouse, or needing combined search + analytics.

---

### Option 4: pgvector for Semantic Search

If exact substring match isn't strictly required, semantic search via embeddings.

```sql
CREATE EXTENSION vector;

CREATE TABLE chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,

    -- Embeddings for semantic search
    input_embedding vector(1536),   -- e.g., OpenAI text-embedding-3-small
    output_embedding vector(1536),

    -- S3 references
    input_s3_key TEXT,
    output_s3_key TEXT,

    -- Metadata
    timestamp TIMESTAMPTZ NOT NULL
);

-- HNSW index for fast similarity search
CREATE INDEX idx_input_embedding ON chat_inference
    USING hnsw (input_embedding vector_cosine_ops);

-- Find semantically similar inferences
SELECT id, function_name,
       1 - (input_embedding <=> $query_embedding) as similarity
FROM chat_inference
ORDER BY input_embedding <=> $query_embedding
LIMIT 10;
```

**Pros:**
- Semantic similarity (more useful than exact substring for many use cases)
- Much smaller index than full-text (1536 floats vs megabytes of text)
- Single database

**Cons:**
- Requires embedding generation (API calls, latency, cost)
- Not exact substring match
- Embeddings need to be regenerated if model changes

**Best for:** Use cases where "find similar" is more valuable than "find exact match".

---

## Comparison Matrix

| Approach | Substring Match | Full Content Search | Complexity | Latency | Additional Services |
|----------|-----------------|---------------------|------------|---------|---------------------|
| pg_trgm (excerpt) | Partial | No | Low | Low | None |
| Elasticsearch | Yes | Yes | High | Medium | Elasticsearch |
| ClickHouse | Yes | Yes | Medium | Low | ClickHouse |
| pgvector | Semantic only | No | Medium | Medium | Embedding API |

## Recommendation

1. **For Postgres-only deployments needing substring search:**
   - Use **Option 1 (pg_trgm with excerpts)** if searching the first portion of content is sufficient
   - Use **Option 2 (Elasticsearch)** if full-content search is required

2. **For deployments that can tolerate ClickHouse:**
   - Use **Option 3 (ClickHouse for search)** - keeps search + analytics together, proven at scale

3. **For semantic/similarity search use cases:**
   - Use **Option 4 (pgvector)** - simpler than Elasticsearch, built into Postgres

## Load Test Considerations

When load testing search performance:
- pg_trgm: Test index build time and query latency with realistic excerpt sizes
- Elasticsearch: Test indexing throughput and search latency separately
- ClickHouse: Test with `position()` and `LIKE` queries on large text columns
