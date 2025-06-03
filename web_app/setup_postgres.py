#!/usr/bin/env python3
"""
Setup script for PostgreSQL persistence in LangGraph web app.

This script helps initialize the PostgreSQL database tables required
for LangGraph checkpointing.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def setup_postgres():
    """Initialize PostgreSQL database for LangGraph persistence."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        
        # Get database URI from environment or use default
        db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")
        
        print(f"🔗 Connecting to PostgreSQL: {db_uri.split('@')[1] if '@' in db_uri else db_uri}")
        
        # Create checkpointer and setup tables
        with PostgresSaver.from_conn_string(db_uri) as checkpointer:
            print("📋 Setting up database tables...")
            checkpointer.setup()
            print("✅ PostgreSQL setup completed successfully!")
            print("\n💡 Your web app is now ready to use PostgreSQL persistence.")
            print("   Conversation history will be preserved across server restarts.")
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install required packages: pip install langgraph-checkpoint-postgres psycopg[binary,pool]")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to setup PostgreSQL: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check your database connection string")
        print("   3. Verify database credentials and permissions")
        print("   4. Ensure the database exists")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 LangGraph PostgreSQL Setup")
    print("=" * 40)
    setup_postgres() 