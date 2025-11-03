#!/usr/bin/env python3
"""
Quick Setup Script for Enhanced NLP RAG Project
Automates the setup process for intelligent source selection
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_python_version():
    """Verify Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def create_directories():
    """Create required project directories"""
    print_header("Creating Project Directories")
    
    directories = [
        "local_knowledge_base",
        "local_knowledge_base/documents",
        "utils",
        "config",
        "tests"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"✅ Created: {directory}/")
        else:
            print(f"✓ Exists: {directory}/")


def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    packages = [
        "duckduckgo-search>=5.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pydantic>=2.0.0",
        "tenacity>=8.2.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0"
    ]
    
    print("Installing new packages...")
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
            return False
    
    return True


def verify_installation():
    """Verify all packages are installed correctly"""
    print_header("Verifying Installation")
    
    required_packages = [
        "duckduckgo_search",
        "requests",
        "bs4",
        "pydantic",
        "tenacity",
        "sentence_transformers",
        "sklearn"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            all_installed = False
    
    return all_installed


def check_env_file():
    """Check if .env file exists and is configured"""
    print_header("Checking Environment Configuration")
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print("⚠️  .env file not found")
        print("\nCreating template .env file...")
        
        with open(".env", "w") as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        
        print("✅ Created .env template")
        print("\n⚠️  IMPORTANT: Edit .env file and add your OpenAI API key!")
        return False
    
    # Check if API key is set
    with open(".env", "r") as f:
        content = f.read()
        if "your_openai_api_key_here" in content or "sk-" not in content:
            print("⚠️  OpenAI API key not configured in .env file")
            print("   Please add your API key to the .env file")
            return False
    
    print("✅ .env file configured")
    return True


def create_test_script():
    """Create a test script to verify setup"""
    print_header("Creating Test Script")
    
    test_script = """#!/usr/bin/env python3
'''
Test script for enhanced RAG components
'''

def test_imports():
    print("Testing imports...")
    try:
        from query_classifier import QueryClassifier
        from web_search_integration import WebSearchIntegration
        from intelligent_source_router import IntelligentSourceRouter
        print("✅ All components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_web_search():
    print("\\nTesting web search...")
    try:
        from web_search_integration import WebSearchIntegration
        
        web_search = WebSearchIntegration(max_results=2)
        results = web_search.search("Python programming")
        
        if results and len(results) > 0:
            print(f"✅ Web search working - Found {len(results)} results")
            print(f"   First result: {results[0]['title'][:50]}...")
            return True
        else:
            print("⚠️  Web search returned no results")
            return False
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return False

def test_query_classifier():
    print("\\nTesting query classifier...")
    try:
        from query_classifier import QueryClassifier
        
        classifier = QueryClassifier()
        result = classifier.classify_query(
            "What's in my document?",
            has_uploaded_docs=True
        )
        
        print(f"✅ Query classifier working")
        print(f"   Test query routed to: {result['datasource']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        return True
    except Exception as e:
        print(f"❌ Query classifier error: {e}")
        return False

def test_router():
    print("\\nTesting intelligent router...")
    try:
        from intelligent_source_router import IntelligentSourceRouter
        
        router = IntelligentSourceRouter()
        result = router.route_query(
            "Test query",
            has_uploaded_docs=False
        )
        
        print(f"✅ Router working")
        print(f"   Routing decision: {result['routing']['datasource']}")
        return True
    except Exception as e:
        print(f"❌ Router error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("  TESTING ENHANCED RAG COMPONENTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Web Search", test_web_search),
        ("Query Classifier", test_query_classifier),
        ("Intelligent Router", test_router)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\\n❌ {name} test failed: {e}")
            results.append(False)
    
    print("\\n" + "="*60)
    print(f"  TEST RESULTS: {sum(results)}/{len(results)} PASSED")
    print("="*60)
    
    if all(results):
        print("\\n🎉 All tests passed! Your setup is ready.")
    else:
        print("\\n⚠️  Some tests failed. Please check the errors above.")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    # Make executable on Unix-like systems
    if os.name != 'nt':
        os.chmod("test_setup.py", 0o755)
    
    print("✅ Created test_setup.py")


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("""
Next Steps:
    
1. Configure your API key:
   - Edit the .env file
   - Add your OpenAI API key: OPENAI_API_KEY=sk-...
   
2. Copy the enhanced files:
   - query_classifier.py
   - web_search_integration.py
   - intelligent_source_router.py
   - enhanced_rag_chatbot.py
   
3. Test the setup:
   python test_setup.py
   
4. Run the enhanced application:
   streamlit run enhanced_rag_chatbot.py
   
5. Read the documentation:
   - See IMPLEMENTATION_GUIDE.md for detailed instructions
   - Check example usage and troubleshooting tips

For help: Review the IMPLEMENTATION_GUIDE.md
    """)


def main():
    """Main setup function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Enhanced NLP RAG Project - Quick Setup Script         ║
║          Intelligent Source Selection & Local RAG            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", lambda: (create_directories(), True)[1]),
        ("Installing dependencies", install_dependencies),
        ("Verifying installation", verify_installation),
        ("Checking environment", check_env_file),
        ("Creating test script", lambda: (create_test_script(), True)[1])
    ]
    
    for step_name, step_func in steps:
        result = step_func()
        if result is False and step_name not in ["Checking environment"]:
            print(f"\n❌ Setup failed at: {step_name}")
            print("   Please resolve the errors and run setup again.")
            sys.exit(1)
    
    print_next_steps()


if __name__ == "__main__":
    main()
