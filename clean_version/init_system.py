#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Initialization Script
Sets up database tables, directories, and core components
"""

import os
import sys
from dotenv import load_dotenv

def create_directories():
    """Create necessary directories"""
    directories = [
        'temp',
        'highlights', 
        'highlights_data',
        'srt_uploads',
        'static/downloads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_environment():
    """Check required environment variables"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file based on .env.example")
        return False
    
    print("✓ Environment variables configured")
    return True

def test_database_connection():
    """Test database connectivity"""
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        
        # Test basic connection
        summaries = db.get_summaries()
        print("✓ Database connection successful")
        print(f"✓ Found {len(summaries) if summaries else 0} existing summaries")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_ai_providers():
    """Test AI provider connections"""
    try:
        from enhanced_summarizer import EnhancedSummarizer
        summarizer = EnhancedSummarizer()
        
        available_providers = summarizer.get_available_providers()
        print(f"✓ AI providers initialized: {available_providers}")
        return True
        
    except Exception as e:
        print(f"❌ AI provider initialization failed: {e}")
        return False

def main():
    """Main initialization routine"""
    print("🚀 Initializing YouTube Summarizer System\n")
    
    # Step 1: Create directories
    print("📁 Creating directories...")
    create_directories()
    
    # Step 2: Check environment
    print("\n🔧 Checking environment...")
    if not check_environment():
        sys.exit(1)
    
    # Step 3: Test database
    print("\n🗄️ Testing database connection...")
    if not test_database_connection():
        print("⚠️ Database connection failed. Application may not work correctly.")
    
    # Step 4: Test AI providers
    print("\n🤖 Testing AI providers...")
    if not test_ai_providers():
        print("⚠️ AI provider initialization failed. Check your API keys.")
    
    print("\n✅ System initialization complete!")
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Visit: http://localhost:5000")
    print("3. Start summarizing videos!")

if __name__ == "__main__":
    main()