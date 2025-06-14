#!/usr/bin/env python3
"""
FAL.ai MCP Server

A Model Context Protocol server for interacting with FAL.ai's generative AI
platform. Provides tools for image generation, video creation, audio
processing, and more.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import httpx

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# MCP imports
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# FAL.ai API configuration
FAL_API_BASE = "https://fal.run"
FAL_QUEUE_BASE = "https://queue.fal.run"


class FALClient:
    """Client for interacting with FAL.ai API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

    async def generate_image(
        self, model: str = "fal-ai/flux/schnell", **kwargs
    ) -> Dict[str, Any]:
        """Generate an image using FAL.ai models"""
        # Use queue for slower models, direct for fast ones
        use_queue = model not in ["fal-ai/flux/schnell"]
        base_url = FAL_QUEUE_BASE if use_queue else FAL_API_BASE

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{base_url}/{model}", headers=self.headers, json=kwargs
            )
            response.raise_for_status()
            return response.json()

    async def generate_video(
        self, model: str = "fal-ai/runway-gen3/turbo/image-to-video", **kwargs
    ) -> Dict[str, Any]:
        """Generate a video using FAL.ai models"""
        # Video generation always uses queue
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{FAL_QUEUE_BASE}/{model}", headers=self.headers, json=kwargs
            )
            response.raise_for_status()
            return response.json()

    async def transcribe_audio(
        self, model: str = "fal-ai/whisper", **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio using FAL.ai models"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{FAL_API_BASE}/{model}", headers=self.headers, json=kwargs
            )
            response.raise_for_status()
            return response.json()

    async def get_queue_status(self, request_url: str) -> Dict[str, Any]:
        """Check status of a queued request"""
        async with httpx.AsyncClient() as client:
            response = await client.get(request_url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def get_queue_result(self, request_url: str) -> Dict[str, Any]:
        """Get result from a queued request"""
        async with httpx.AsyncClient() as client:
            response = await client.get(request_url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def cancel_request(self, request_url: str) -> Dict[str, Any]:
        """Cancel a queued request"""
        async with httpx.AsyncClient() as client:
            response = await client.delete(request_url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload a file to FAL.ai storage"""
        async with aiofiles.open(file_path, "rb") as f:
            file_content = await f.read()

        # First, get upload URL
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FAL_API_BASE}/storage/upload/initiate",
                headers=self.headers,
                json={
                    "content_type": self._get_content_type(file_path),
                    "file_name": Path(file_path).name,
                },
            )
            response.raise_for_status()
            upload_data = response.json()

        # Upload file
        async with httpx.AsyncClient() as client:
            response = await client.put(
                upload_data["upload_url"],
                content=file_content,
                headers={"Content-Type": self._get_content_type(file_path)},
            )
            response.raise_for_status()

        return {"url": upload_data["file_url"]}

    def _get_content_type(self, file_path: str) -> str:
        """Get content type for file"""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".txt": "text/plain",
            ".json": "application/json",
        }
        return content_types.get(ext, "application/octet-stream")


# Initialize the MCP server
server = Server("fal-ai-mcp")

# Global FAL client
fal_client: Optional[FALClient] = None


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="generate_image",
            description="Generate images using FAL.ai models",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the image to generate",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for generation",
                        "default": "fal-ai/flux/schnell",
                        "enum": [
                            "fal-ai/flux/schnell",
                            "fal-ai/flux/dev",
                            "fal-ai/flux-pro/v1.1-ultra",
                            "fal-ai/stable-diffusion-v35-large",
                            "fal-ai/recraft/v3/text-to-image",
                            "fal-ai/hidream-i1-full",
                        ],
                    },
                    "image_size": {
                        "type": "string",
                        "description": "Size of the generated image",
                        "default": "landscape_4_3",
                        "enum": [
                            "square_hd",
                            "square",
                            "portrait_4_3",
                            "portrait_16_9",
                            "landscape_4_3",
                            "landscape_16_9",
                        ],
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of inference steps",
                        "default": 28,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "Guidance scale for generation",
                        "default": 3.5,
                        "minimum": 1.0,
                        "maximum": 20.0,
                    },
                    "num_images": {
                        "type": "integer",
                        "description": "Number of images to generate",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 4,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="generate_video",
            description="Generate videos from text or images using FAL.ai models",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the video to generate",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for video generation",
                        "default": "fal-ai/runway-gen3/turbo/image-to-video",
                        "enum": [
                            "fal-ai/runway-gen3/turbo/image-to-video",
                            "fal-ai/runway-gen3/turbo/text-to-video",
                            "fal-ai/kling-video/v2/master/image-to-video",
                            "fal-ai/kling-video/v2/master/text-to-video",
                            "fal-ai/minimax/video-01/image-to-video",
                        ],
                    },
                    "image_url": {
                        "type": "string",
                        "description": "URL of input image for image-to-video models",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration of video in seconds",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="transcribe_audio",
            description="Transcribe audio to text using Whisper",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_url": {
                        "type": "string",
                        "description": "URL of the audio file to transcribe",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task type",
                        "default": "transcribe",
                        "enum": ["transcribe", "translate"],
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "Language of the audio (optional, auto-detected if not specified)"
                        ),
                    },
                },
                "required": ["audio_url"],
            },
        ),
        Tool(
            name="upload_file",
            description="Upload a file to FAL.ai storage for use in other operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Local path to the file to upload",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="check_queue_status",
            description="Check the status of a queued FAL.ai request",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_url": {
                        "type": "string",
                        "description": "The request URL returned from a queued operation",
                    }
                },
                "required": ["request_url"],
            },
        ),
        Tool(
            name="get_queue_result",
            description="Get the result of a completed queued FAL.ai request",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_url": {
                        "type": "string",
                        "description": "The request URL returned from a queued operation",
                    }
                },
                "required": ["request_url"],
            },
        ),
        Tool(
            name="cancel_request",
            description="Cancel a queued FAL.ai request",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_url": {
                        "type": "string",
                        "description": "The request URL of the request to cancel",
                    }
                },
                "required": ["request_url"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    if not fal_client:
        return [
            TextContent(type="text", text="Error: FAL_KEY environment variable not set")
        ]

    try:
        if name == "generate_image":
            # Extract parameters
            prompt = arguments["prompt"]
            model = arguments.get("model", "fal-ai/flux/schnell")

            # Build parameters based on model
            params = {"prompt": prompt}

            # Add common parameters
            if "image_size" in arguments:
                params["image_size"] = arguments["image_size"]
            if "num_inference_steps" in arguments:
                params["num_inference_steps"] = arguments["num_inference_steps"]
            if "guidance_scale" in arguments:
                params["guidance_scale"] = arguments["guidance_scale"]
            if "num_images" in arguments:
                params["num_images"] = arguments["num_images"]
            if "seed" in arguments:
                params["seed"] = arguments["seed"]

            result = await fal_client.generate_image(model=model, **params)

            # Format response
            if "request_url" in result:
                response_text = (
                    f"Image generation queued!\n\nRequest URL: {result['request_url']}\n\n"
                    f"Use check_queue_status to monitor progress."
                )
            else:
                response_text = (
                    f"Image generated successfully!\n\n"
                    f"{json.dumps(result, indent=2)}"
                )

            return [TextContent(type="text", text=response_text)]

        elif name == "generate_video":
            prompt = arguments["prompt"]
            model = arguments.get("model", "fal-ai/runway-gen3/turbo/image-to-video")

            params = {"prompt": prompt}

            if "image_url" in arguments:
                params["image_url"] = arguments["image_url"]
            if "duration" in arguments:
                params["duration"] = arguments["duration"]

            result = await fal_client.generate_video(model=model, **params)

            response_text = (
                f"Video generation queued!\n\nRequest URL: {result['request_url']}\n\n"
                f"Use check_queue_status to monitor progress."
            )
            return [TextContent(type="text", text=response_text)]

        elif name == "transcribe_audio":
            audio_url = arguments["audio_url"]

            params = {"audio_url": audio_url}
            if "task" in arguments:
                params["task"] = arguments["task"]
            if "language" in arguments:
                params["language"] = arguments["language"]

            result = await fal_client.transcribe_audio(**params)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "upload_file":
            file_path = arguments["file_path"]
            if not os.path.exists(file_path):
                return [
                    TextContent(type="text", text=f"Error: File not found: {file_path}")
                ]

            result = await fal_client.upload_file(file_path)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "check_queue_status":
            result = await fal_client.get_queue_status(arguments["request_url"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_queue_result":
            result = await fal_client.get_queue_result(arguments["request_url"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "cancel_request":
            result = await fal_client.cancel_request(arguments["request_url"])
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point"""
    global fal_client

    # Get API key from environment
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        print("Error: FAL_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    # Initialize FAL client
    fal_client = FALClient(api_key)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fal-ai-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
