# Postgres Load Test for TensorZero

This load test simulates the database writes that occur during inference requests in a Postgres-only TensorZero deployment.

Supports:
- Local Postgres
- PlanetScale (Postgres mode)
- Any Postgres-compatible database

## What It Tests

Each simulated inference request performs **4 writes in a single transaction**:

| Table | Operation | Data Size | Purpose |
|-------|-----------|-----------|---------|
| `chat_inference` | INSERT | ~3KB | Store inference input/output |
| `model_inference` | INSERT | ~3KB | Store model request/response |
| `resource_bucket` | UPSERT | ~100B | Rate limiting state |
| `variant_by_episode` | UPSERT | ~100B | Experimentation tracking |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- PostgreSQL running locally (or accessible via URL)
- Default connection: `postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest`

## Quick Start

```bash
cd scripts/postgres_load_test

# 1. Create the test database
psql postgresql://postgres:postgres@localhost:5432/postgres -c "CREATE DATABASE tensorzero_loadtest;"

# 2. Set up the schema
psql postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest -f schema.sql

# 3. Run a baseline test (50 RPS for 60 seconds)
uv run load_test.py --rps 50 --duration 60 --workers 4
```

## Usage

```bash
uv run load_test.py [OPTIONS]

Options:
  --postgres-url TEXT       Postgres connection URL
                            [default: postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest]
  --rps INTEGER             Target requests per second [default: 100]
  --duration INTEGER        Test duration in seconds [default: 60]
  --workers INTEGER         Number of worker processes [default: 8]
  --output-size-kb FLOAT    Size of output JSON in KB [default: 2048 (2MB)]
  --raw-response-size-kb    Size of raw_response JSON in KB [default: 2048 (2MB)]
  --input-size-kb FLOAT     Size of input JSON in KB [default: 1.5]
  --raw-request-size-kb     Size of raw_request JSON in KB [default: 1.0]
  --no-rate-limiting        Skip rate limiting writes
  --no-experimentation      Skip experimentation writes
  --docker-container TEXT   Docker container name to monitor CPU/memory
  --server-side-data        Generate random data in SQL (eliminates network transfer)
  --verbose                 Print verbose error messages
  --help                    Show this message and exit
```

## PlanetScale Setup

1. Create a PlanetScale database with Postgres mode enabled
2. Get the connection string from the PlanetScale dashboard
3. Create the schema:

```bash
# Using the PlanetScale connection string
psql "postgres://user:password@host.connect.psdb.cloud/database?sslmode=require" -f schema.sql

# Or for higher throughput testing, use unlogged tables (if supported)
psql "postgres://user:password@host.connect.psdb.cloud/database?sslmode=require" -f schema_unlogged.sql
```

4. Run the load test:

```bash
uv run load_test.py \
  --postgres-url "postgres://user:password@host.connect.psdb.cloud/database?sslmode=require" \
  --rps 1000 \
  --duration 60 \
  --workers 16 \
  --output-size-kb 1024 \
  --raw-response-size-kb 1024
```

**Notes for PlanetScale:**
- PlanetScale is serverless, so connection pooling behavior may differ
- Network latency will be higher than local Postgres
- Start with fewer workers (8-16) and increase gradually
- UNLOGGED tables may not be supported; use regular `schema.sql`

## Test Scenarios

| Scenario | Command | Purpose |
|----------|---------|---------|
| Baseline | `uv run load_test.py --rps 50 --duration 60 --workers 4` | Establish baseline |
| Medium | `uv run load_test.py --rps 200 --duration 120 --workers 8` | Normal production |
| High | `uv run load_test.py --rps 500 --duration 120 --workers 16` | Peak traffic |
| Stress | `uv run load_test.py --rps 1000 --duration 60 --workers 32` | Find limits |
| Sustained | `uv run load_test.py --rps 200 --duration 600 --workers 8` | Long-term stability |
| Large Output | `uv run load_test.py --rps 100 --duration 60 --workers 8 --output-size-kb 1024 --raw-response-size-kb 1024` | 1MB outputs |
| Server-Side | `uv run load_test.py --rps 1000 --duration 60 --workers 32 --server-side-data` | Pure DB throughput |

### Server-Side Data Generation

Use `--server-side-data` to generate random data using SQL functions instead of sending data from the client. This eliminates network transfer overhead and measures pure database write throughput.

This is especially useful when:
- Testing remote databases (PlanetScale, RDS, etc.) where network is a bottleneck
- Benchmarking database write performance independent of client location
- Running high-throughput tests where network bandwidth is limited

```bash
# Test PlanetScale write throughput without network bottleneck
uv run load_test.py \
  --postgres-url "postgres://user:pass@host.connect.psdb.cloud/database?sslmode=require" \
  --rps 1000 \
  --duration 60 \
  --workers 32 \
  --output-size-kb 1024 \
  --raw-response-size-kb 1024 \
  --server-side-data
```

## Example Output

```
============================================================
POSTGRES LOAD TEST FOR TENSORZERO
============================================================
Target RPS:          100
Duration:            60s
Connections:         10
Rate limiting:       enabled
Experimentation:     enabled
============================================================

Starting workers...
  Progress: 5s elapsed, 498 requests, 99.6 RPS
  Progress: 10s elapsed, 1,001 requests, 100.1 RPS
  ...

============================================================
LOAD TEST RESULTS
============================================================

THROUGHPUT:
  Total Requests:     6,000
  Successful Writes:  6,000
  Failed Writes:      0
  Actual RPS:         100.0
  Success Rate:       100.00%

LATENCY (ms):
  Min:                1.23
  Avg:                3.45
  P50:                2.89
  P95:                6.12
  P99:                9.87
  Max:                15.43

DATA WRITTEN:
  Estimated:          35.2 MB

============================================================
STATUS: PASSED - All writes successful, target RPS achieved
============================================================
```

## Metrics to Monitor

### From the Script

- **Throughput**: Actual RPS achieved, success rate
- **Latency**: P50, P95, P99, min/max/avg (ms)

### External Monitoring (optional)

Monitor Postgres during the test:

```bash
# Watch active connections
watch -n 1 'psql postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest -c "SELECT count(*) FROM pg_stat_activity WHERE datname = '\''tensorzero_loadtest'\'';"'

# Check table sizes after test
psql postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest -c "
SELECT
    relname as table,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    n_live_tup as rows
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
"
```

## Reset Between Tests

To clear all data and start fresh:

```bash
psql postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest -c "SELECT reset_loadtest_tables();"
```

Or drop and recreate:

```bash
psql postgresql://postgres:postgres@localhost:5432/postgres -c "DROP DATABASE IF EXISTS tensorzero_loadtest;"
psql postgresql://postgres:postgres@localhost:5432/postgres -c "CREATE DATABASE tensorzero_loadtest;"
psql postgresql://postgres:postgres@localhost:5432/tensorzero_loadtest -f schema.sql
```

## Benchmark Results (Local Postgres in Docker)

Results from testing on a local Postgres instance with UNLOGGED tables:

| Output Size | Total/Request | P50 Latency | Max RPS | Bottleneck |
|-------------|---------------|-------------|---------|------------|
| 10 KB | 22 KB | 2.3 ms | 690+ RPS | Client-limited |
| 1 MB | 2 MB | 87 ms | ~80 RPS | Postgres CPU |
| 2 MB | 4 MB | 134 ms | ~45 RPS | Postgres CPU |

**Key findings:**
- With small data (22KB), Postgres easily handles 700+ RPS
- With 1-2MB outputs, Postgres CPU becomes the bottleneck (TOAST compression, JSONB processing)
- Adding more workers beyond the optimal point increases latency due to contention
- UNLOGGED tables provide ~2x improvement over regular tables

**Postgres tuning for high write throughput:**
```bash
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:16 \
  -c shared_buffers=256MB \
  -c wal_buffers=64MB \
  -c max_wal_size=2GB \
  -c synchronous_commit=off \
  -c max_connections=200
```

## Distributed Testing (Multiple VMs)

For high-throughput testing, run the load test across multiple VMs using the coordinator script.

### Prerequisites

1. SSH access to all VMs (key-based auth recommended)
2. Install load test on each VM:
   ```bash
   # On each VM
   git clone <repo> ~/postgres_load_test
   cd ~/postgres_load_test
   # Or copy files via scp
   ```
3. Ensure `uv` is installed on each VM

### Running Distributed Tests

```bash
# Run across 4 VMs, 250 RPS each = 1000 RPS total
uv run coordinator.py \
  --vms "vm1.example.com,vm2.example.com,vm3.example.com,vm4.example.com" \
  --postgres-url "postgres://user:pass@host:5432/db?sslmode=require" \
  --rps 250 \
  --duration 60 \
  --workers 8 \
  --output-size-kb 1024 \
  --raw-response-size-kb 1024

# With SSH options
uv run coordinator.py \
  --vms "10.0.0.1,10.0.0.2,10.0.0.3" \
  --ssh-user ubuntu \
  --ssh-key ~/.ssh/loadtest.pem \
  --remote-path ~/postgres_load_test \
  --postgres-url "YOUR_URL" \
  --rps 333 \
  --duration 60
```

### Coordinator Options

```bash
Options:
  --vms TEXT                Comma-separated list of VM hostnames or IPs (required)
  --postgres-url TEXT       Postgres connection URL (required)
  --rps INTEGER             Target RPS per VM [default: 100]
  --duration INTEGER        Test duration in seconds [default: 60]
  --workers INTEGER         Workers per VM [default: 8]
  --output-size-kb FLOAT    Output size in KB [default: 1024]
  --raw-response-size-kb    Raw response size in KB [default: 1024]
  --ssh-user TEXT           SSH username (default: current user)
  --ssh-key TEXT            Path to SSH private key
  --remote-path TEXT        Path to load test on remote VMs [default: ~/postgres_load_test]
  --json-output             Output aggregated results as JSON
```

### Single-VM JSON Output

For manual aggregation or custom coordination:

```bash
uv run load_test.py --json-output --rps 100 --duration 60 > results.json
```

## Files

- `load_test.py` - Main load test script (uses multiprocessing for parallelism)
- `coordinator.py` - Distributed test coordinator (runs tests across multiple VMs via SSH)
- `schema.sql` - Database schema for test tables (regular tables)
- `schema_unlogged.sql` - Schema with UNLOGGED tables (faster, no durability)
- `pyproject.toml` - Python project configuration (dependencies managed by uv)
- `README.md` - This file
