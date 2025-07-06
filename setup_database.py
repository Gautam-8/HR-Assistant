#!/usr/bin/env python3
"""
Database setup script for HR Knowledge Assistant
This script creates the PostgreSQL database and sets up the pgvector extension.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
from config import config

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            database='postgres'  # Connect to default database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.POSTGRES_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database: {config.POSTGRES_DB}")
            cursor.execute(f"CREATE DATABASE {config.POSTGRES_DB}")
            print("Database created successfully!")
        else:
            print(f"Database {config.POSTGRES_DB} already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    
    return True

def setup_pgvector():
    """Setup pgvector extension"""
    try:
        # Connect to the HR database
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            database=config.POSTGRES_DB
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create pgvector extension
        print("Setting up pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector extension created successfully!")
        
        # Verify extension
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print("pgvector extension is active")
        else:
            print("Warning: pgvector extension not found")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error setting up pgvector: {e}")
        return False
    
    return True

def test_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            database=config.POSTGRES_DB
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        if version:
            print(f"PostgreSQL version: {version[0]}")
        else:
            print("Could not retrieve PostgreSQL version")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("HR Knowledge Assistant - Database Setup")
    print("=" * 50)
    
    print(f"Database configuration:")
    print(f"  Host: {config.POSTGRES_HOST}")
    print(f"  Port: {config.POSTGRES_PORT}")
    print(f"  Database: {config.POSTGRES_DB}")
    print(f"  User: {config.POSTGRES_USER}")
    print()
    
    # Step 1: Create database
    if not create_database():
        print("Failed to create database. Exiting.")
        sys.exit(1)
    
    # Step 2: Setup pgvector
    if not setup_pgvector():
        print("Failed to setup pgvector. Exiting.")
        sys.exit(1)
    
    # Step 3: Test connection
    if not test_connection():
        print("Connection test failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Database setup completed successfully!")
    print("You can now run the HR Knowledge Assistant.")
    print("=" * 50)

if __name__ == "__main__":
    main() 