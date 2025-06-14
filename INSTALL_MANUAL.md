# Manual Installation Guide

Since you encountered the "externally-managed-environment" error, here's the manual setup process:

## 1. Create Virtual Environment

```bash
cd /Users/ajm/Documents/Code/2025/fal-mcp
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Set API Key

Choose one of these methods:

**Method 1: .env file (recommended)**
```bash
cp .env.example .env
# Edit .env file with your actual API key
```

**Method 2: Export in terminal**
```bash
export FAL_KEY="your-fal-api-key-here"
```

**Method 3: Add to shell profile (persistent)**
```bash
echo 'export FAL_KEY="your-fal-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## 4. Test the Server

```bash
python test_server.py
```

## 5. Run the Server

```bash
# For testing
python main.py

# Or with MCP Inspector
npx @modelcontextprotocol/inspector python main.py
```

## 6. Configure Claude Desktop

Edit your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "fal-ai": {
      "command": "/Users/ajm/Documents/Code/2025/fal-mcp/venv/bin/python",
      "args": ["/Users/ajm/Documents/Code/2025/fal-mcp/main.py"],
      "env": {
        "FAL_KEY": "your-fal-api-key-here"
      }
    }
  }
}
```

## 7. Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

## What the Error Meant

The error you saw is Python's new safety feature (PEP 668) that prevents installing packages globally on system-managed Python installations. Using a virtual environment is the recommended solution and what we've implemented above.

## Next Steps

Once you've completed the manual setup:

1. Test image generation: "Generate a sunset landscape"
2. Try video creation: "Turn this image into a video"
3. Test transcription: "Transcribe this audio file"

The server is fully functional and ready to use!
