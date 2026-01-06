# TensorZero: ClickHouse to Postgres-Only Migration Plan

## Goal

Enable new TensorZero deployments to run with Postgres-only (no ClickHouse dependency), while maintaining full feature parity.

## Scope

- **New deployments only** - no dual-write or backfill of existing ClickHouse data
- **Phased migration** with feature flags - flip flag when all tables are migrated
- **Mirrored schema** - keep separate tables like ClickHouse (ChatInference/JsonInference, etc.)
- **Indexes over materialized views** - use proper B-tree/GIN indexes instead of secondary lookup tables
- **Background worker for stats** - aggregate statistics periodically instead of on-write
- **Start with inferences** - core write path first

---

## Phase 1: Infrastructure (Foundation)

### 1.1 Add Backend Selection to Config

**File: `tensorzero-core/src/config/mod.rs`**

```rust
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityBackend {
    #[default]
    Clickhouse,
    Postgres,
}

pub struct ObservabilityConfig {
    // ... existing fields ...
    #[serde(default)]
    pub backend: ObservabilityBackend,
}
```

### 1.2 Create ObservabilityBackend Enum

**New file: `tensorzero-core/src/db/observability_backend.rs`**

```rust
pub enum ObservabilityBackend {
    ClickHouse(ClickHouseConnectionInfo),
    Postgres(PgPool),
    Disabled,
}
```

Implement delegate pattern for all existing traits:

- `SelectQueries`, `DatasetQueries`, `FeedbackQueries`
- `InferenceQueries`, `ModelInferenceQueries`
- `ConfigQueries`, `EvaluationQueries`
- `HealthCheckable`

### 1.3 Update AppStateData

**File: `tensorzero-core/src/utils/gateway.rs`**

Replace `clickhouse_connection_info: ClickHouseConnectionInfo` with:

```rust
pub observability_backend: ObservabilityBackend,
```

Keep `postgres_connection_info: PostgresConnectionInfo` for rate limiting/experimentation.

### 1.4 UUIDv7 Timestamp Extraction

**New file: `tensorzero-core/src/utils/uuidv7.rs`**

Create utility to extract timestamp from UUIDv7:

```rust
pub fn timestamp_from_uuidv7(id: Uuid) -> DateTime<Utc> {
    let bytes = id.as_bytes();
    let timestamp_ms = u64::from_be_bytes([0, 0, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]]);
    DateTime::from_timestamp_millis(timestamp_ms as i64).unwrap_or_default()
}
```

---

## Phase 2: Postgres Schema (Separate Tables - Mirroring ClickHouse)

**New migration: `tensorzero-core/src/db/postgres/migrations/2025XXXX_observability_tables.sql`**

### 2.1 Chat Inference Table

```sql
CREATE TABLE chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    tool_params JSONB,
    inference_params JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    ttft_ms INTEGER,
    dynamic_tools TEXT[],
    dynamic_provider_tools TEXT[],
    allowed_tools JSONB,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL  -- Extracted from UUIDv7 in application
);

CREATE INDEX idx_chat_inference_function_variant_episode ON chat_inference(function_name, variant_name, episode_id);
CREATE INDEX idx_chat_inference_episode ON chat_inference(episode_id);
CREATE INDEX idx_chat_inference_timestamp ON chat_inference(timestamp DESC);
CREATE INDEX idx_chat_inference_tags ON chat_inference USING GIN(tags);
```

### 2.2 JSON Inference Table

```sql
CREATE TABLE json_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    output_schema JSONB NOT NULL,
    inference_params JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER,
    tags JSONB NOT NULL DEFAULT '{}',
    extra_body JSONB,
    auxiliary_content TEXT,
    ttft_ms INTEGER,
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_json_inference_function_variant_episode ON json_inference(function_name, variant_name, episode_id);
CREATE INDEX idx_json_inference_episode ON json_inference(episode_id);
CREATE INDEX idx_json_inference_timestamp ON json_inference(timestamp DESC);
CREATE INDEX idx_json_inference_tags ON json_inference USING GIN(tags);
```

### 2.3 Model Inference Table

```sql
CREATE TABLE model_inference (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,  -- References chat_inference or json_inference
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    response_time_ms INTEGER,
    ttft_ms INTEGER,
    system TEXT,
    input_messages JSONB NOT NULL,
    output JSONB NOT NULL,
    cached BOOLEAN NOT NULL DEFAULT FALSE,
    finish_reason TEXT,
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_model_inference_inference_id ON model_inference(inference_id);
CREATE INDEX idx_model_inference_model ON model_inference(model_name, model_provider_name);
CREATE INDEX idx_model_inference_timestamp ON model_inference(timestamp DESC);
```

### 2.4 Feedback Tables (Separate by Type)

```sql
-- Boolean metric feedback
CREATE TABLE boolean_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value BOOLEAN NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_boolean_feedback_metric_target ON boolean_metric_feedback(metric_name, target_id);
CREATE INDEX idx_boolean_feedback_target ON boolean_metric_feedback(target_id);

-- Float metric feedback
CREATE TABLE float_metric_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_float_feedback_metric_target ON float_metric_feedback(metric_name, target_id);
CREATE INDEX idx_float_feedback_target ON float_metric_feedback(target_id);

-- Comment feedback
CREATE TABLE comment_feedback (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    target_type TEXT NOT NULL CHECK (target_type IN ('inference', 'episode')),
    value TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_comment_feedback_target ON comment_feedback(target_id);

-- Demonstration feedback
CREATE TABLE demonstration_feedback (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,
    value TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    snapshot_hash BYTEA,
    timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_demonstration_feedback_inference ON demonstration_feedback(inference_id);
```

### 2.5 Datapoint Tables (Separate by Type)

```sql
-- Chat inference datapoint
CREATE TABLE chat_inference_datapoint (
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    id UUID NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    tool_params JSONB,
    tags JSONB NOT NULL DEFAULT '{}',
    auxiliary JSONB NOT NULL DEFAULT '{}',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    staled_at TIMESTAMPTZ,
    source_inference_id UUID,
    is_custom BOOLEAN NOT NULL DEFAULT FALSE,
    name TEXT,
    dynamic_tools TEXT[],
    dynamic_provider_tools TEXT[],
    allowed_tools JSONB,
    tool_choice TEXT,
    parallel_tool_calls BOOLEAN,
    snapshot_hash BYTEA,
    PRIMARY KEY (dataset_name, function_name, id)
);
CREATE INDEX idx_chat_datapoint_id ON chat_inference_datapoint(id);
CREATE INDEX idx_chat_datapoint_updated ON chat_inference_datapoint(updated_at DESC);

-- JSON inference datapoint
CREATE TABLE json_inference_datapoint (
    dataset_name TEXT NOT NULL,
    function_name TEXT NOT NULL,
    id UUID NOT NULL,
    episode_id UUID,
    input JSONB NOT NULL,
    output JSONB,
    output_schema JSONB NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}',
    auxiliary JSONB NOT NULL DEFAULT '{}',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    staled_at TIMESTAMPTZ,
    source_inference_id UUID,
    is_custom BOOLEAN NOT NULL DEFAULT FALSE,
    name TEXT,
    snapshot_hash BYTEA,
    PRIMARY KEY (dataset_name, function_name, id)
);
CREATE INDEX idx_json_datapoint_id ON json_inference_datapoint(id);
CREATE INDEX idx_json_datapoint_updated ON json_inference_datapoint(updated_at DESC);
```

### 2.6 Statistics Tables (for analytics)

```sql
-- Feedback statistics (aggregated by background job)
CREATE TABLE feedback_variant_statistics (
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    count BIGINT NOT NULL DEFAULT 0,
    sum_value DOUBLE PRECISION NOT NULL DEFAULT 0,
    sum_squared DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (function_name, metric_name, variant_name, period_start)
);

-- Model provider statistics
CREATE TABLE model_provider_statistics (
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    count BIGINT NOT NULL DEFAULT 0,
    total_input_tokens BIGINT NOT NULL DEFAULT 0,
    total_output_tokens BIGINT NOT NULL DEFAULT 0,
    sum_response_time_ms BIGINT NOT NULL DEFAULT 0,
    sum_ttft_ms BIGINT NOT NULL DEFAULT 0,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_name, model_provider_name, period_start)
);

CREATE TABLE cumulative_usage (
    type TEXT PRIMARY KEY,
    count BIGINT NOT NULL DEFAULT 0
);
```

### 2.7 Additional Tables

```sql
-- Batch tables
CREATE TABLE batch_request (...);
CREATE TABLE batch_model_inference (...);

-- Evaluation tables
CREATE TABLE dynamic_evaluation_run (...);
CREATE TABLE dynamic_evaluation_run_episode (...);
CREATE TABLE static_evaluation_human_feedback (...);

-- Config tables
CREATE TABLE config_snapshot (...);
CREATE TABLE tensorzero_migration_pg (...);
CREATE TABLE deployment_id (...);

-- Cache
CREATE TABLE model_inference_cache (...);

-- Dynamic in-context learning
CREATE TABLE dynamic_in_context_learning_example (...);
```

---

## Phase 3: Implement Postgres Query Traits

### 3.1 Inference Queries

**New file: `tensorzero-core/src/db/postgres/inference_queries.rs`**

Implement `InferenceQueries` trait for `PgPool`:

- `list_inferences()` - with pagination, filtering
- `count_inferences()`
- `get_inference_by_id()`

### 3.2 Model Inference Queries

**New file: `tensorzero-core/src/db/postgres/model_inference_queries.rs`**

Implement `ModelInferenceQueries` trait.

### 3.3 Feedback Queries

**New file: `tensorzero-core/src/db/postgres/feedback_queries.rs`**

Implement `FeedbackQueries` trait.

### 3.4 Dataset Queries

**New file: `tensorzero-core/src/db/postgres/dataset_queries.rs`**

Implement `DatasetQueries` trait.

### 3.5 Select Queries (Analytics)

**New file: `tensorzero-core/src/db/postgres/select_queries.rs`**

Implement `SelectQueries` trait:

- `count_distinct_models_used()`
- `get_model_usage_timeseries()`
- `get_model_latency_quantiles()` - use `percentile_cont`
- `query_episode_table()`

---

## Phase 4: Update Write Paths

### 4.1 Inference Write Path

**File: `tensorzero-core/src/endpoints/inference.rs`**

Modify `write_to_database()` to dispatch based on backend:

```rust
match &app_state.observability_backend {
    ObservabilityBackend::ClickHouse(ch) => {
        ch.write_batched(...).await?;
    }
    ObservabilityBackend::Postgres(pool) => {
        sqlx::query!("INSERT INTO inference ...").execute(pool).await?;
    }
    ObservabilityBackend::Disabled => {}
}
```

### 4.2 Feedback Write Path

**File: `tensorzero-core/src/endpoints/feedback/mod.rs`**

Update all feedback write functions.

### 4.3 Dataset Write Path

**File: `tensorzero-core/src/endpoints/datasets/v1/*.rs`**

Update datapoint CRUD operations.

---

## Phase 5: Statistics Aggregation

### 5.1 Background Aggregation Task

Create a Tokio task that periodically aggregates statistics:

- Run every minute
- Aggregate feedback by variant into `feedback_variant_statistics`
- Aggregate model metrics into `model_provider_statistics`
- Update `cumulative_usage` counters

**New file: `tensorzero-core/src/db/postgres/statistics_worker.rs`**

---

## Phase 6: Testing

### 6.1 Unit Tests

Add tests for each new Postgres query implementation.

### 6.2 Integration Tests

Add E2E tests that run with `backend = "postgres"`.

### 6.3 Parity Tests

Create tests that verify identical behavior between ClickHouse and Postgres backends.

---

## Implementation Order

| Step | Description                                          | Dependencies |
| ---- | ---------------------------------------------------- | ------------ |
| 1    | Add `ObservabilityBackend` enum to config            | None         |
| 2    | Create `observability_backend.rs` with delegate pattern | Step 1       |
| 3    | Create UUIDv7 timestamp utility                      | None         |
| 4    | Create Postgres migration with all tables            | Steps 1-3    |
| 5    | Update `AppStateData` to use new backend             | Steps 1-2    |
| 6    | Implement `InferenceQueries` for Postgres            | Step 4       |
| 7    | Update inference write path                          | Steps 5-6    |
| 8    | Implement `FeedbackQueries` for Postgres             | Step 4       |
| 9    | Update feedback write path                           | Steps 5, 8   |
| 10   | Implement `DatasetQueries` for Postgres              | Step 4       |
| 11   | Update dataset write path                            | Steps 5, 10  |
| 12   | Implement `SelectQueries` for Postgres               | Step 4       |
| 13   | Create statistics aggregation worker                 | Step 4       |
| 14   | Add batch/evaluation support                         | Steps 5-6    |
| 15   | Add E2E tests                                        | All above    |

---

## Critical Files Summary

### Config Changes

- `tensorzero-core/src/config/mod.rs` - Add `ObservabilityBackend` enum

### New Files

- `tensorzero-core/src/db/observability_backend.rs` - Backend abstraction
- `tensorzero-core/src/db/postgres/inference_queries.rs`
- `tensorzero-core/src/db/postgres/model_inference_queries.rs`
- `tensorzero-core/src/db/postgres/feedback_queries.rs`
- `tensorzero-core/src/db/postgres/dataset_queries.rs`
- `tensorzero-core/src/db/postgres/select_queries.rs`
- `tensorzero-core/src/db/postgres/statistics_worker.rs`
- `tensorzero-core/src/db/postgres/migrations/2025XXXX_observability_tables.sql`
- `tensorzero-core/src/utils/uuidv7.rs`

### Modified Files

- `tensorzero-core/src/utils/gateway.rs` - Update `AppStateData`
- `tensorzero-core/src/db/mod.rs` - Add new trait bounds
- `tensorzero-core/src/endpoints/inference.rs` - Dispatch to backend
- `tensorzero-core/src/endpoints/feedback/mod.rs` - Dispatch to backend
- `tensorzero-core/src/endpoints/datasets/v1/*.rs` - Dispatch to backend

---

## ClickHouse Features to Postgres Equivalents

| ClickHouse Feature           | Postgres Equivalent                                        |
| ---------------------------- | ---------------------------------------------------------- |
| `UUIDv7ToDateTime()`         | Application-level extraction                               |
| `MaterializedViews`          | B-tree + GIN indexes (no secondary tables needed)          |
| `ReplacingMergeTree`         | Standard UPSERT (`ON CONFLICT`)                            |
| `AggregatingMergeTree`       | Background aggregation worker                              |
| `SummingMergeTree`           | `ON CONFLICT DO UPDATE SET count = count + 1`              |
| `LowCardinality(String)`     | Not needed (optimizer handles)                             |
| `Map(String, String)`        | `JSONB`                                                    |
| `Array(String)`              | Native `TEXT[]` arrays                                     |
| Bloom filter index           | B-tree + GIN indexes                                       |
| Separate Chat/Json tables    | Keep separate (same structure)                             |
