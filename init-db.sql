-- Initialize database for LangGraph checkpointing
-- The langgraph-checkpoint-postgres package will automatically create the necessary tables
-- when setup() is called, so we just need to ensure the database exists and basic settings

-- Set timezone
SET timezone = 'UTC';

-- Create extension for UUID generation if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant necessary permissions (optional, but good for security)
GRANT ALL PRIVILEGES ON DATABASE langgraph TO postgres;