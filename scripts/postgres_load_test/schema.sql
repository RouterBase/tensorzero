-- Postgres Load Test Schema for TensorZero
-- This creates simplified tables that mirror the structure needed for inference writes

-- Chat inference table (simplified from full schema)
CREATE TABLE IF NOT EXISTS chat_inference (
    id UUID PRIMARY KEY,
    function_name TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    inference_params JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '{}',
    processing_time_ms INTEGER,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_inference_function_variant
    ON chat_inference(function_name, variant_name);
CREATE INDEX IF NOT EXISTS idx_chat_inference_episode
    ON chat_inference(episode_id);
CREATE INDEX IF NOT EXISTS idx_chat_inference_timestamp
    ON chat_inference(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_chat_inference_tags
    ON chat_inference USING GIN(tags);

-- Model inference table (simplified from full schema)
CREATE TABLE IF NOT EXISTS model_inference (
    id UUID PRIMARY KEY,
    inference_id UUID NOT NULL,
    raw_request TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider_name TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    response_time_ms INTEGER,
    input_messages JSONB NOT NULL,
    output JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_inference_inference_id
    ON model_inference(inference_id);
CREATE INDEX IF NOT EXISTS idx_model_inference_model
    ON model_inference(model_name, model_provider_name);
CREATE INDEX IF NOT EXISTS idx_model_inference_timestamp
    ON model_inference(timestamp DESC);

-- Rate limiting table (from existing TensorZero schema)
CREATE TABLE IF NOT EXISTS resource_bucket (
    key TEXT PRIMARY KEY,
    tickets BIGINT NOT NULL,
    balance_as_of TIMESTAMPTZ NOT NULL
);

-- Experimentation table (from existing TensorZero schema)
CREATE TABLE IF NOT EXISTS variant_by_episode (
    function_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    variant_name TEXT NOT NULL,
    PRIMARY KEY (function_name, episode_id)
);

-- Utility: Reset all tables for a fresh test run
-- Usage: SELECT reset_loadtest_tables();
CREATE OR REPLACE FUNCTION reset_loadtest_tables() RETURNS void AS $$
BEGIN
    TRUNCATE chat_inference, model_inference, resource_bucket, variant_by_episode;
END;
$$ LANGUAGE plpgsql;
