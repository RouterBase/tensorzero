-- Postgres Load Test Schema for TensorZero (UNLOGGED - faster, no durability)
-- Use this for benchmarking only. Data is lost on crash/restart.

-- Drop existing tables first
DROP TABLE IF EXISTS chat_inference CASCADE;
DROP TABLE IF EXISTS model_inference CASCADE;
DROP TABLE IF EXISTS resource_bucket CASCADE;
DROP TABLE IF EXISTS variant_by_episode CASCADE;

-- Chat inference table (UNLOGGED for speed)
CREATE UNLOGGED TABLE chat_inference (
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

-- Minimal indexes for load testing (fewer indexes = faster writes)
CREATE INDEX idx_chat_inference_timestamp ON chat_inference(timestamp DESC);

-- Model inference table (UNLOGGED for speed)
CREATE UNLOGGED TABLE model_inference (
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

CREATE INDEX idx_model_inference_timestamp ON model_inference(timestamp DESC);

-- Rate limiting table (UNLOGGED)
CREATE UNLOGGED TABLE resource_bucket (
    key TEXT PRIMARY KEY,
    tickets BIGINT NOT NULL,
    balance_as_of TIMESTAMPTZ NOT NULL
);

-- Experimentation table (UNLOGGED)
CREATE UNLOGGED TABLE variant_by_episode (
    function_name TEXT NOT NULL,
    episode_id UUID NOT NULL,
    variant_name TEXT NOT NULL,
    PRIMARY KEY (function_name, episode_id)
);

-- Utility: Reset all tables for a fresh test run
CREATE OR REPLACE FUNCTION reset_loadtest_tables() RETURNS void AS $$
BEGIN
    TRUNCATE chat_inference, model_inference, resource_bucket, variant_by_episode;
END;
$$ LANGUAGE plpgsql;
