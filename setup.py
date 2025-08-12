#!/usr/bin/env python3
"""
YouTube Summarizer - Complete Setup Script
Guides users through initial system setup
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} - OK")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file with template"""
    env_path = Path(".env")
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    env_template = """# YouTube Summarizer Configuration
# Copy this template and fill in your actual values

# Required: Supabase Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# Required: AI Services
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Additional AI Providers
ELEVENLABS_API_KEY=your_elevenlabs_key_here
CLAUDE_API_KEY=your_claude_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GEMINI_API_KEY=your_gemini_key_here

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_DEBUG=false
PORT=5000

# Database Selection
USE_SUPABASE=true
"""
    
    try:
        with open(env_path, "w") as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("📝 Edit .env with your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found")
        print("   Video processing will be limited")
        print("   Install from: https://ffmpeg.org/download.html")
        return False

def database_setup_instructions():
    """Provide database setup instructions"""
    print("\n📋 Database Setup Instructions:")
    print("1. Create a Supabase project at https://supabase.com")
    print("2. Go to Settings > API to get your URL and service role key")
    print("3. In the SQL Editor, run: sql/supabase_vector_search.sql")
    print("4. Verify with: sql/health_check.sql")
    print("5. Update your .env file with the credentials")

def main():
    """Main setup flow"""
    print_header("YouTube Summarizer Setup")
    
    print("This script will help you set up the YouTube Summarizer system.")
    print("Make sure you have:")
    print("• Python 3.8+")
    print("• pip (Python package manager)")
    print("• A Supabase account")
    print("• OpenAI API access")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check Python
    print_step(1, "Checking Python Version")
    if not check_python_version():
        return False
    
    # Step 2: Install dependencies
    print_step(2, "Installing Dependencies")
    if not install_dependencies():
        return False
    
    # Step 3: Create environment file
    print_step(3, "Setting up Environment")
    create_env_file()
    
    # Step 4: Check FFmpeg
    print_step(4, "Checking FFmpeg")
    check_ffmpeg()
    
    # Step 5: Database setup
    print_step(5, "Database Setup")
    database_setup_instructions()
    
    # Final instructions
    print_header("Setup Complete!")
    print("🎉 Basic setup is done!")
    print("\nNext steps:")
    print("1. Edit .env with your actual API keys")
    print("2. Set up your Supabase database (see instructions above)")
    print("3. Run: python init.py (to verify setup)")
    print("4. Run: python app.py (to start the application)")
    
    print("\n💡 Need help? Check README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)