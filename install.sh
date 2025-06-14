#!/bin/bash

# FAL.ai MCP Server Installation Script
# This script helps you set up the FAL.ai MCP server quickly

set -e

echo "🚀 FAL.ai MCP Server Installation"
echo "=================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is not compatible. Please install Python 3.10 or higher."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
echo "   Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Check for FAL_KEY
echo "🔑 Checking for FAL.ai API key..."
if [ -z "$FAL_KEY" ] && [ ! -f ".env" ]; then
    echo "⚠️  FAL_KEY not found in environment or .env file."
    echo "   Please get your API key from: https://fal.ai/dashboard"
    echo "   Then choose one method:"
    echo "   • Create .env file: cp .env.example .env (recommended)"
    echo "   • Export in terminal: export FAL_KEY='your-api-key-here'"
    echo "   • Add to shell profile for persistence"
else
    echo "✅ FAL_KEY configuration found"
fi

# Run tests
echo "🧪 Running tests..."
source venv/bin/activate
python test_server.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Set your FAL_KEY if you haven't already:"
echo "   # Method 1: Create .env file (recommended)"
echo "   cp .env.example .env"
echo "   # Edit .env with your actual API key"
echo ""
echo "   # Method 2: Export in terminal"
echo "   export FAL_KEY='your-fal-api-key-here'"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Test the server:"
echo "   python main.py"
echo ""
echo "4. Or use MCP Inspector for interactive testing:"
echo "   npx @modelcontextprotocol/inspector python main.py"
echo ""
echo "5. Add to Claude Desktop (see README.md for details):"
echo "   - Edit your claude_desktop_config.json"
echo "   - Add the server configuration"
echo "   - Restart Claude Desktop"
echo ""
echo "6. Try the examples in examples.py!"
echo ""
echo "Happy generating! 🎨🎬🎵"
