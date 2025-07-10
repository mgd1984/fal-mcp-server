# FAL.ai MCP Server

> *Because manually copy-pasting prompts into web UIs is so 2023.*

A Model Context Protocol (MCP) server that brings FAL.ai's generative AI superpowers directly into your AI assistant workflow. Generate images, create videos, transcribe audio, and more—all without leaving your conversation.

## Features

- **Image Generation**: FLUX, Stable Diffusion, and other cutting-edge models
- **Video Creation**: Image-to-video and text-to-video generation
- **Audio Transcription**: Whisper-powered speech-to-text
- **File Management**: Upload files to FAL.ai CDN
- **Queue Management**: Handle long-running tasks asynchronously
- **Model Discovery**: Search and explore available models
- **Schema Introspection**: Get detailed model parameters

## Quick Start

### One-Click Setup with Codename [Goose](goose://recipe?config=eyJpZCI6InVudGl0bGVkIiwibmFtZSI6IlVudGl0bGVkIFJlY2lwZSIsInRpdGxlIjoiRkFMLmFpIFBhdGNoIEdlbmVyYXRvciBSZWNpcGUiLCJkZXNjcmlwdGlvbiI6IkEgc2FtcGxlIEdvb3NlIHJlY2lwZSBmb3IgdXNpbmcgdGhlIEZBTC5haSBNQ1AgdG9vbCBzZXJ2ZXIiLCJwYXJhbWV0ZXJzIjpbXSwiaW5zdHJ1Y3Rpb25zIjoiQXNzaXN0IGluIGNyZWF0aW5nIGRldGFpbGVkIGFuZCBjb25jZXB0dWFsbHkgcmljaCBsb2dvIGFuZCBwYXRjaCBkZXNpZ25zIGZvciBicmFuZHMgb3IgY29tcGFuaWVzLCBmb2N1c2luZyBvbiBtaW5pbWFsaXN0aWMgYW5kIHN5bWJvbGljIHZpc3VhbCBlbGVtZW50cy4gVXNlIEFJIGltYWdlIGdlbmVyYXRpb24gdG9vbHMgdG8gY3JhZnQgdW5pcXVlIGJyYW5kaW5nIGVsZW1lbnRzIGxpa2UgZW1ibGVtcyBvciBiYWRnZXMgdGhhdCByZXByZXNlbnQgY29tcGxleCBpZGVhcyBhbmQgbWV0YXBob3JzLCB3aGlsZSBhZGhlcmluZyB0byBzcGVjaWZpYyBzdHlsZSByZXF1ZXN0cyBzdWNoIGFzIGVtYnJvaWRlcnkgc3VpdGFiaWxpdHksIGFzcGVjdCByYXRpb3MsIGNvbG9yIHBhbGV0dGVzLCBhbmQgaWNvbm9ncmFwaGljIHNpbXBsaWNpdHkuIEVuc3VyZSBvdXRwdXRzIGNvbW11bmljYXRlIHRoZSByZXF1ZXN0ZWQgc3ltYm9saXNtIGNsZWFybHkgYW5kIG1haW50YWluIGEgY29uc2lzdGVudCBzdHlsZSBzdWl0ZWQgZm9yIHBoeXNpY2FsIG1lcmNoYW5kaXNlIG9yIGRpZ2l0YWwgYnJhbmRpbmcuIiwiYWN0aXZpdGllcyI6WyJHZW5lcmF0ZSBtaW5pbWFsaXN0aWMgbG9nbyBkZXNpZ25zIiwiQ3JlYXRlIHN5bWJvbGljIHBhdGNoIGVtYmxlbXMiLCJEZXNpZ24gZW1icm9pZGVyZWQgYmlrZXIgYmFkZ2VzIiwiVmlzdWFsaXplIGVzb3RlcmljIGFuZCBvY2N1bHQgbW90aWZzIiwiRGV2ZWxvcCBicmFuZCBpY29ub2dyYXBoeSJdLCJwcm9tcHQiOiIiLCJleHRlbnNpb25zIjpbXX0%3D)

<img src="goose.png" alt="Codename Goose" width="64" height="64" style="display:block;margin:0 auto 16px;">


### Using the Recipe Generator
Visit the [Goose Recipe Generator](https://block.github.io/goose/recipes) to create custom recipes.

</details>

---




### Prerequisites

- Python 3.10+
- A FAL.ai API key ([get one here](https://fal.ai/dashboard))

### Installation

```bash
# Clone and install
git clone <your-repo-url>
cd fal-mcp

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key (choose one method)
# Method 1: Create .env file (recommended)
cp .env.example .env
# Edit .env with your actual API key

# Method 2: Export in terminal
export FAL_KEY="your-fal-api-key-here"
```

**Quick install**: Just run `./install.sh` and it handles everything!

### Running the Server

**Development mode** (with MCP Inspector):
```bash
# Activate virtual environment first
source venv/bin/activate
npx @modelcontextprotocol/inspector python main.py
```

**Production mode**:
```bash
# Activate virtual environment first
source venv/bin/activate
python main.py
```

### Cursor Integration

```json
{
  "mcpServers": {
    "fal-ai": {
      "command": "/path/to/your/fal-mcp/venv/bin/python",
      "args": ["/path/to/your/fal-mcp/main.py"],
      "env": {
        "FAL_KEY": "your-fal-api-key-here"
      }
    }
  }
}
```

### Claude Desktop Integration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fal-ai": {
      "command": "/path/to/your/fal-mcp/venv/bin/python",
      "args": ["/path/to/your/fal-mcp/main.py"],
      "env": {
        "FAL_KEY": "your-fal-api-key-here"
      }
    }
  }
}
```

## Usage Examples

### Generate Images

```
User: Generate a cyberpunk cityscape at sunset

AI: I'll create that image for you using FLUX...
[Calls generate_image with prompt="cyberpunk cityscape at sunset"]

Generated image: https://fal.media/files/elephant/xyz123.png
```

### Create Videos from Images

```
User: Turn this image into a video with gentle camera movement

AI: I'll animate that image for you...
[Calls generate_video with the image URL and motion prompt]

Video generation queued. I'll check the status...
[Calls check_queue_status]

Your video is ready: https://fal.media/files/tiger/abc456.mp4
```

### Transcribe Audio

```
User: Transcribe this podcast episode for me

AI: I'll transcribe that audio file...
[Calls transcribe_audio with the audio URL]

Transcription complete! Here's what I found:
"Welcome to the AI podcast where we discuss..."
```

## Available Tools

### Core Generation Tools

- `generate_image` - Create images with FLUX, Stable Diffusion, Recraft, HiDream, etc.
- `generate_video` - Generate videos from text or images using Runway Gen3, Kling, MiniMax
- `transcribe_audio` - Convert speech to text with Whisper
- `generate_speech` - Generate speech from text using MiniMax Speech 2.0 HD
- `vectorize_image` - Vectorize images with Recraft

### File & Queue Operations

- `upload_file` - Upload files to FAL.ai storage
- `check_queue_status` - Monitor long-running tasks
- `get_queue_result` - Retrieve completed results
- `cancel_request` - Stop queued operations

### Model Management

- `list_models` - List all available models
- `search_models` - Search for models by name
- `get_model_schema` - Get the schema for a specific model

## Configuration

### Environment Variables

The server needs your FAL.ai API key. Get yours from the [FAL.ai dashboard](https://fal.ai/dashboard/keys).

**Option 1: .env file (recommended for local development)**
```bash
# Copy the example and edit with your key
cp .env.example .env
# Edit .env: FAL_KEY=your_actual_api_key_here
```

**Option 2: Export in terminal**
```bash
export FAL_KEY="your-api-key-here"
```

**Option 3: Claude Desktop config**
```json
{
  "mcpServers": {
    "fal-ai": {
      "env": {
        "FAL_KEY": "your-api-key-here"
      }
    }
  }
}
```

**Required:**
- `FAL_KEY` - Your FAL.ai API key

### Model Defaults

The server uses sensible defaults but everything is configurable:

- **Image model**: `fal-ai/flux/schnell` (fast, high-quality)
- **Video model**: `fal-ai/runway-gen3/turbo/image-to-video`
- **Audio model**: `fal-ai/whisper`
- **Image size**: `landscape_4_3`
- **Inference steps**: 28
- **Guidance scale**: 3.5

## Advanced Usage

### Queue Management for Long Tasks

Video generation and complex image tasks can take time. The server automatically handles queuing:

```python
# Generate video (automatically queued)
result = await generate_video(
    prompt="A serene lake with gentle ripples",
    image_url="https://example.com/lake.jpg",
    duration=10,
    queue=True  # Default for video
)

# Check status periodically
status = await check_queue_status(result["request_url"])

# Get final result when ready
final_result = await get_queue_result(result["request_url"])
```

### File Upload Workflow

```python
# Upload local file first
upload_result = await upload_file("/path/to/image.jpg")
image_url = upload_result["url"]

# Use in generation
video_result = await generate_video(
    prompt="Add magical sparkles",
    image_url=image_url
)
```

## Error Handling

The server provides detailed error messages:

- **Missing API key**: Clear setup instructions
- **Invalid model**: Suggestions for similar models
- **File not found**: Path validation and suggestions
- **API errors**: Formatted FAL.ai error responses

## Development

### Project Structure

```
fal-mcp/
├── main.py           # MCP server implementation
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

### Adding New Models

The server is designed to work with any FAL.ai model. To add support for new model types:

1. Add a new tool definition in `handle_list_tools()`
2. Implement the handler in `handle_call_tool()`
3. Update the README with usage examples

### Testing

```bash
# Test with MCP Inspector
npx @modelcontextprotocol/inspector python main.py

# Test individual tools
python -c "
import asyncio
from main import FALClient
client = FALClient('your-key')
result = asyncio.run(client.list_models())
print(result)
"
```

## Troubleshooting

### Common Issues

**"FAL_KEY environment variable not set"**
- Create a `.env` file: `cp .env.example .env` and edit with your key
- Or export in terminal: `export FAL_KEY="your-key"`
- Or add to your shell profile for persistence

**"Model not found"**
- Use `list_models` or `search_models` to find available models
- Check the FAL.ai documentation for model IDs

**"File upload failed"**
- Verify file exists and is readable
- Check file size limits (varies by model)
- Ensure stable internet connection

**Queue requests timing out**
- Use `check_queue_status` to monitor progress
- Some models take 5-10 minutes for complex generations
- Consider using smaller parameters for faster results

### Getting Help

- Check the [FAL.ai documentation](https://docs.fal.ai/)
- Review model-specific requirements and limits
- Test with the MCP Inspector for debugging

## License

MIT License - build cool stuff with it.

## Contributing

Found a bug? Want to add support for new FAL.ai models? PRs welcome!

---
