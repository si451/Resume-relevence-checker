#!/usr/bin/env python3
"""
Setup script to configure the Groq API for the Resume Relevance Checker.
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with Groq API configuration."""
    env_content = """# Groq API Configuration
GROK_API_URL=https://api.groq.com/openai/v1
GROK_API_KEY=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

# Optional: Override default settings
# HARD_WEIGHT=0.6
# SOFT_WEIGHT=0.4
# EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return False
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    print("🔑 Using the provided Groq API key")
    print("🚀 You can now run the application with: streamlit run streamlit_app.py")
    return True

def test_api_connection():
    """Test the API connection."""
    try:
        from scoring.grok_client import check_grok_health
        print("🔍 Testing API connection...")
        health = check_grok_health()
        
        if health["ok"]:
            print("✅ API connection successful!")
            return True
        else:
            print(f"❌ API connection failed: {health.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

def main():
    """Main setup function."""
    print("🔧 Resume Relevance Checker - API Setup")
    print("=" * 50)
    
    # Create .env file
    if create_env_file():
        # Test API connection
        print("\n🧪 Testing API connection...")
        if test_api_connection():
            print("\n🎉 Setup completed successfully!")
            print("You can now run the application with:")
            print("  streamlit run streamlit_app.py")
        else:
            print("\n⚠️  Setup completed but API test failed.")
            print("The application will use fallback methods for AI features.")
    else:
        print("\n❌ Setup failed.")

if __name__ == "__main__":
    main()
