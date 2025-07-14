#!/bin/bash

# Reset local PostgreSQL database for LangGraph checkpointer
# This script will drop and recreate the checkpoint tables

set -e  # Exit on any error

# Local PostgreSQL connection settings (matching setup_postgres.py)
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="postgres"
DB_USER="aubrey"
DB_PASSWORD="postgres"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 Resetting LangGraph PostgreSQL checkpointer (port 5432)...${NC}"

# Set password for psql
export PGPASSWORD="$DB_PASSWORD"

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
    echo -e "${RED}❌ PostgreSQL is not running on ${DB_HOST}:${DB_PORT}${NC}"
    echo -e "${BLUE}💡 Please start your PostgreSQL service on port 5432 first${NC}"
    exit 1
fi

echo -e "${BLUE}✅ PostgreSQL is running on ${DB_HOST}:${DB_PORT}${NC}"

# Drop existing checkpoint tables
echo -e "${YELLOW}📋 Dropping existing checkpoint tables...${NC}"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
DROP TABLE IF EXISTS checkpoints CASCADE;
DROP TABLE IF EXISTS checkpoint_writes CASCADE;
DROP TABLE IF EXISTS checkpoint_blobs CASCADE;
DROP TABLE IF EXISTS checkpoint_migrations CASCADE;
DROP TABLE IF EXISTS writes CASCADE;
" >/dev/null 2>&1

echo -e "${GREEN}✅ Checkpoint tables dropped successfully${NC}"
echo -e "${GREEN}✅ Database reset complete!${NC}"