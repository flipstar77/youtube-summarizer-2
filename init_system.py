#!/usr/bin/env python3
"""
YouTube Summarizer - System Initialization & Health Check
Living checkpoint for setting up and verifying the complete system
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SystemInitializer:
    """Initialize and verify YouTube Summarizer system"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_total = 0
        self.errors = []
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "[INFO]",
            "SUCCESS": "[OK]", 
            "WARNING": "[WARN]",
            "ERROR": "[ERROR]"
        }.get(level, "[LOG]")
        print(f"[{timestamp}] {prefix} {message}")
        
    def check(self, name: str, func, critical: bool = True) -> bool:
        """Run a check and track results"""
        self.checks_total += 1
        self.log(f"Checking {name}...")
        
        try:
            result = func()
            if result:
                self.checks_passed += 1
                self.log(f"{name}: OK", "SUCCESS")
                return True
            else:
                msg = f"{name}: FAILED"
                if critical:
                    self.errors.append(msg)
                    self.log(msg, "ERROR")
                else:
                    self.warnings.append(msg)
                    self.log(msg, "WARNING")
                return False
        except Exception as e:
            msg = f"{name}: ERROR - {str(e)}"
            if critical:
                self.errors.append(msg)
                self.log(msg, "ERROR")
            else:
                self.warnings.append(msg)
                self.log(msg, "WARNING")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables"""
        required_vars = [
            ("SUPABASE_URL", True),
            ("SUPABASE_SERVICE_ROLE_KEY", True),
            ("OPENAI_API_KEY", True),
        ]
        
        optional_vars = [
            ("ELEVENLABS_API_KEY", False),
            ("CLAUDE_API_KEY", False),
            ("PERPLEXITY_API_KEY", False),
        ]
        
        missing_required = []
        missing_optional = []
        
        for var, required in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var, _ in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_required:
            self.log(f"Missing required env vars: {', '.join(missing_required)}", "ERROR")
            return False
        
        if missing_optional:
            self.log(f"Missing optional env vars: {', '.join(missing_optional)}", "WARNING")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies"""
        required_packages = [
            "flask", "openai", "supabase", "python-dotenv", 
            "yt-dlp", "youtube-transcript-api", "elevenlabs",
            "requests", "ffmpeg-python", "moviepy", "APScheduler"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            self.log(f"Missing packages: {', '.join(missing)}", "ERROR")
            self.log("Run: pip install -r requirements.txt", "INFO")
            return False
        
        return True
    
    def check_database_files(self) -> bool:
        """Check database schema files exist"""
        required_files = [
            "sql/supabase_vector_search.sql",
            "sql/health_check.sql"
        ]
        
        missing = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing.append(file_path)
        
        if missing:
            self.log(f"Missing database files: {', '.join(missing)}", "ERROR")
            return False
        
        return True
    
    def check_supabase_connection(self) -> bool:
        """Test Supabase connection"""
        if not SUPABASE_AVAILABLE:
            self.log("Supabase client not available", "ERROR")
            return False
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            return False
        
        try:
            client = create_client(url, key)
            # Test connection with simple query
            result = client.table("summaries").select("id").limit(1).execute()
            return True
        except Exception as e:
            self.log(f"Supabase connection failed: {str(e)}", "ERROR")
            return False
    
    def check_vector_setup(self) -> bool:
        """Check vector search setup"""
        if not SUPABASE_AVAILABLE:
            return False
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        try:
            client = create_client(url, key)
            
            # Check vector extension
            result = client.rpc("sql", {
                "query": "SELECT 1 FROM pg_extension WHERE extname='vector'"
            }).execute()
            
            if not result.data:
                self.log("pgvector extension not found", "ERROR")
                return False
            
            # Check embedding column
            result = client.table("summaries").select("embedding").limit(1).execute()
            
            # Test RPC functions
            dummy_vector = [0.0] * 1536
            client.rpc("search_summaries_by_similarity", {
                "query_embedding": dummy_vector,
                "match_threshold": 0.0,
                "match_count": 1
            }).execute()
            
            return True
            
        except Exception as e:
            self.log(f"Vector setup check failed: {str(e)}", "ERROR")
            return False
    
    def check_openai_connection(self) -> bool:
        """Test OpenAI API connection"""
        if not OPENAI_AVAILABLE:
            self.log("OpenAI client not available", "ERROR")
            return False
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False
        
        try:
            client = openai.OpenAI(api_key=api_key)
            # Test with simple models list
            client.models.list()
            return True
        except Exception as e:
            self.log(f"OpenAI connection failed: {str(e)}", "ERROR")
            return False
    
    def check_ffmpeg(self) -> bool:
        """Check FFmpeg availability"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log("FFmpeg not found - video processing will fail", "WARNING")
            return False
    
    def check_project_structure(self) -> bool:
        """Check project directory structure"""
        required_dirs = [
            "templates", "static", "sql", "tests"
        ]
        
        required_files = [
            "app.py", "database.py", "supabase_client.py",
            "enhanced_summarizer.py", "vector_embeddings.py",
            "automation_scheduler.py", "requirements.txt"
        ]
        
        missing_dirs = [d for d in required_dirs if not Path(d).exists()]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_dirs or missing_files:
            if missing_dirs:
                self.log(f"Missing directories: {', '.join(missing_dirs)}", "ERROR")
            if missing_files:
                self.log(f"Missing files: {', '.join(missing_files)}", "ERROR")
            return False
        
        return True
    
    def run_quick_functional_test(self) -> bool:
        """Run a quick end-to-end test"""
        try:
            # Test imports
            from enhanced_summarizer import EnhancedSummarizer
            from vector_embeddings import SummaryVectorizer
            from supabase_client import SupabaseDatabase
            
            # Test basic initialization
            db = SupabaseDatabase()
            vectorizer = SummaryVectorizer()
            summarizer = EnhancedSummarizer()
            
            self.log("Core components initialized successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Functional test failed: {str(e)}", "ERROR")
            return False
    
    def setup_recommendations(self):
        """Provide setup recommendations based on check results"""
        self.log("\n" + "="*60, "INFO")
        self.log("SETUP RECOMMENDATIONS", "INFO")
        self.log("="*60, "INFO")
        
        if self.errors:
            self.log("CRITICAL ISSUES TO FIX:", "ERROR")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
            
            self.log("\nNEXT STEPS:", "INFO")
            self.log("1. Fix environment variables in .env file", "INFO")
            self.log("2. Install missing dependencies: pip install -r requirements.txt", "INFO")
            self.log("3. Apply database schema: sql/supabase_vector_search.sql", "INFO")
            self.log("4. Run health check: sql/health_check.sql", "INFO")
        
        if self.warnings:
            self.log("\nOPTIONAL IMPROVEMENTS:", "WARNING")
            for warning in self.warnings:
                self.log(f"  - {warning}", "WARNING")
        
        if not self.errors:
            self.log("SYSTEM READY!", "SUCCESS")
            self.log("You can start the application with: python app.py", "INFO")
    
    def run_full_initialization(self):
        """Run complete system initialization check"""
        self.log("YouTube Summarizer - System Initialization", "INFO")
        self.log(f"Starting health check at {datetime.now()}", "INFO")
        self.log("-" * 60, "INFO")
        
        # Critical checks
        self.check("Environment Variables", self.check_environment_variables, critical=True)
        self.check("Python Dependencies", self.check_dependencies, critical=True)
        self.check("Project Structure", self.check_project_structure, critical=True)
        self.check("Database Files", self.check_database_files, critical=True)
        self.check("Supabase Connection", self.check_supabase_connection, critical=True)
        self.check("Vector Search Setup", self.check_vector_setup, critical=True)
        self.check("OpenAI Connection", self.check_openai_connection, critical=True)
        
        # Optional checks
        self.check("FFmpeg", self.check_ffmpeg, critical=False)
        self.check("Functional Test", self.run_quick_functional_test, critical=False)
        
        # Summary
        self.log("-" * 60, "INFO")
        self.log(f"Health Check Complete: {self.checks_passed}/{self.checks_total} checks passed", "INFO")
        
        self.setup_recommendations()
        
        return len(self.errors) == 0

def main():
    """Main initialization entry point"""
    initializer = SystemInitializer()
    success = initializer.run_full_initialization()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()