#!/usr/bin/env python3
"""
FAL.ai MCP Server Usage Examples

This file shows example interactions with the FAL.ai MCP server.
These are the types of conversations you can have with Claude Desktop
once the MCP server is configured.
"""

# Example 1: Image Generation
"""
User: Generate a cyberpunk cityscape at sunset with neon lights

AI: I'll create that image for you using FLUX...
[Calls generate_image tool with:
{
  "prompt": "cyberpunk cityscape at sunset with neon lights",
  "model": "fal-ai/flux/schnell",
  "image_size": "landscape_4_3",
  "num_inference_steps": 28,
  "guidance_scale": 3.5
}]

Generated image: https://fal.media/files/elephant/xyz123.png

The image shows a stunning cyberpunk cityscape with towering skyscrapers
illuminated by vibrant neon lights against a dramatic sunset sky.
"""

# Example 2: Video Generation from Image
"""
User: Turn this image into a video with gentle camera movement
[User provides image URL]

AI: I'll animate that image for you with gentle camera movement...
[Calls generate_video tool with:
{
  "prompt": "gentle camera movement, cinematic pan",
  "image_url": "https://example.com/image.jpg",
  "duration": 5,
  "queue": true
}]

Video generation has been queued. Let me check the status...
[Calls check_queue_status tool]

Your video is processing. This typically takes 2-3 minutes for a 5-second video.
I'll check again in a moment...

[After checking status again]
Your video is ready! https://fal.media/files/tiger/abc456.mp4
"""

# Example 3: Audio Transcription
"""
User: Transcribe this podcast episode for me
[User provides audio URL]

AI: I'll transcribe that audio file using Whisper...
[Calls transcribe_audio tool with:
{
  "audio_url": "https://example.com/podcast.mp3",
  "task": "transcribe",
  "language": "en"
}]

Transcription complete! Here's what I found:

"Welcome to the AI podcast where we discuss the latest developments
in artificial intelligence. Today we're talking about the future of
generative AI and its impact on creative industries..."

[Full transcription continues...]

Would you like me to summarize the key points or create timestamps?
"""

# Example 4: Model Discovery
"""
User: What image generation models are available?

AI: Let me check the available image generation models for you...
[Calls search_models tool with:
{
  "keywords": "image generation"
}]

Here are the available image generation models:

1. **fal-ai/flux/schnell** - Fast, high-quality image generation
2. **fal-ai/stable-diffusion-v3-medium** - Stable Diffusion v3
3. **fal-ai/stable-diffusion-xl** - SDXL for detailed images
4. **fal-ai/aura-flow** - Artistic style generation

Would you like me to generate an image with any of these models?
"""

# Example 5: File Upload and Processing
"""
User: I have a local image I want to animate. It's at /Users/me/photo.jpg

AI: I'll upload your image and then animate it for you...
[Calls upload_file tool with:
{
  "file_path": "/Users/me/photo.jpg"
}]

Image uploaded successfully: https://fal.media/files/uploaded/photo123.jpg

Now I'll create a video animation from your image...
[Calls generate_video tool with the uploaded image URL]

Your animated video is being generated. This will take a few minutes...
"""

# Example 6: Queue Management
"""
User: Cancel that video generation, I want to try different settings

AI: I'll cancel the current video generation for you...
[Calls cancel_request tool with the queue URL]

Video generation cancelled successfully. The queue request has been stopped.

What different settings would you like to try? I can:
- Adjust the duration (1-10 seconds)
- Change the motion prompt
- Use a different model
- Modify other parameters
"""

# Example 7: Model Schema Exploration
"""
User: What parameters can I use with the FLUX model?

AI: Let me get the detailed schema for the FLUX model...
[Calls get_model_schema tool with:
{
  "model_id": "fal-ai/flux/schnell"
}]

Here are the available parameters for FLUX Schnell:

**Required:**
- prompt (string): Text description of the image

**Optional:**
- image_size: square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3,
  landscape_16_9
- num_inference_steps: 1-50 (default: 28)
- guidance_scale: 1.0-20.0 (default: 3.5)
- num_images: 1-4 (default: 1)
- seed: Random seed for reproducible results

Would you like me to generate an image with specific parameters?
"""

if __name__ == "__main__":
    print("FAL.ai MCP Server Examples")
    print("=" * 30)
    print("This file contains example interactions with the FAL.ai MCP server.")
    print("Configure the server in Claude Desktop to try these examples!")
    print("\nSee README.md for setup instructions.")
