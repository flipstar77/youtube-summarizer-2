#!/usr/bin/env python3
"""
YouTube Summarizer - Quick System Check
Run this to verify your setup is ready
"""

if __name__ == "__main__":
    try:
        from init_system import SystemInitializer
        
        print("YouTube Summarizer - Quick System Check")
        print("=" * 50)
        
        initializer = SystemInitializer()
        success = initializer.run_full_initialization()
        
        if success:
            print("\nSUCCESS: System is ready! Start with: python app.py")
        else:
            print("\nERROR: Setup incomplete. See recommendations above.")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Try: pip install -r requirements.txt")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Check your environment and try again")