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
        # Clean up parameters based on model
        clean_params = {}

        # Common parameters
        if "prompt" in kwargs:
            clean_params["prompt"] = kwargs["prompt"]
        if "negative_prompt" in kwargs:
            clean_params["negative_prompt"] = kwargs["negative_prompt"]
        if "seed" in kwargs:
            clean_params["seed"] = kwargs["seed"]
        if "num_images" in kwargs:
            clean_params["num_images"] = kwargs["num_images"]

        # Model-specific parameter handling
        if "imagen4" in model or "kontext" in model:
            # Imagen4 and Kontext use aspect_ratio instead of image_size
            if "aspect_ratio" in kwargs:
                clean_params["aspect_ratio"] = kwargs["aspect_ratio"]
            elif "image_size" in kwargs:
                # Convert image_size to aspect_ratio
                size_map = {
                    "square_hd": "1:1",
                    "square": "1:1",
                    "portrait_4_3": "3:4",
                    "portrait_16_9": "9:16",
                    "landscape_4_3": "4:3",
                    "landscape_16_9": "16:9",
                }
                clean_params["aspect_ratio"] = size_map.get(kwargs["image_size"], "1:1")

            # Kontext-specific parameters
            if "kontext" in model:
                if "safety_tolerance" in kwargs:
                    clean_params["safety_tolerance"] = kwargs["safety_tolerance"]
                if "output_format" in kwargs:
                    clean_params["output_format"] = kwargs["output_format"]
                if "sync_mode" in kwargs:
                    clean_params["sync_mode"] = kwargs["sync_mode"]

                # Kontext Max requires image_url for image-to-image editing
                if "kontext/max" in model and "image_url" in kwargs:
                    clean_params["image_url"] = kwargs["image_url"]
        elif "recraft" in model:
            # Recraft uses style and image_size
            if "style" in kwargs:
                clean_params["style"] = kwargs["style"]
            if "image_size" in kwargs:
                clean_params["image_size"] = kwargs["image_size"]
        elif "hidream" in model:
            # HiDream uses image_size and guidance_scale
            if "image_size" in kwargs:
                clean_params["image_size"] = kwargs["image_size"]
            if "guidance_scale" in kwargs:
                clean_params["guidance_scale"] = kwargs["guidance_scale"]
            if "num_inference_steps" in kwargs:
                clean_params["num_inference_steps"] = kwargs["num_inference_steps"]
        else:
            # FLUX models use image_size and num_inference_steps
            if "image_size" in kwargs:
                clean_params["image_size"] = kwargs["image_size"]
            if "num_inference_steps" in kwargs:
                clean_params["num_inference_steps"] = kwargs["num_inference_steps"]

        # Use queue for slower models, direct for fast ones
        use_queue = model not in ["fal-ai/flux/schnell", "fal-ai/imagen4/preview/fast"]
        base_url = FAL_QUEUE_BASE if use_queue else FAL_API_BASE

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{base_url}/{model}", headers=self.headers, json=clean_params
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

    def get_available_tools(self) -> List[Tool]:
        """Return list of available MCP tools."""
        return [
            Tool(
                name="generate_image",
                description="Generate images using FAL.ai models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the image to generate",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use for generation",
                            "enum": [
                                "fal-ai/flux/schnell",
                                "fal-ai/flux/dev",
                                "fal-ai/imagen4/preview/fast",
                                "fal-ai/recraft-v3",
                                "fal-ai/hidream-i1-full",
                                "fal-ai/imagen4/preview",
                                "fal-ai/imagen4/preview/ultra",
                            ],
                            "default": "fal-ai/flux/schnell",
                        },
                        "image_size": {
                            "type": "string",
                            "description": "Size of the generated image",
                            "enum": [
                                "square_hd",
                                "square",
                                "portrait_4_3",
                                "portrait_16_9",
                                "landscape_4_3",
                                "landscape_16_9",
                                "1:1",
                                "16:9",
                                "9:16",
                                "3:4",
                                "4:3",
                            ],
                            "default": "square_hd",
                        },
                        "num_images": {
                            "type": "integer",
                            "description": "Number of images to generate (1-4)",
                            "minimum": 1,
                            "maximum": 4,
                            "default": 1,
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "What to avoid in the generated image",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducible results",
                        },
                        "guidance_scale": {
                            "type": "number",
                            "description": "How closely to follow the prompt (1-20)",
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "num_inference_steps": {
                            "type": "integer",
                            "description": "Number of denoising steps (1-50)",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "style": {
                            "type": "string",
                            "description": "Style for Recraft models",
                            "enum": [
                                "realistic_image",
                                "digital_illustration",
                                "vector_illustration",
                            ],
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Imagen4 models",
                            "enum": ["1:1", "16:9", "9:16", "3:4", "4:3"],
                        },
                    },
                    "required": ["prompt"],
                },
            ),
            Tool(
                name="vectorize_image",
                description="Convert raster images to SVG format using Recraft",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": (
                                "URL of the image to vectorize. Must be PNG, JPG or WEBP, "
                                "less than 5MB, resolution less than 16MP, max dimension "
                                "less than 4096px, min dimension more than 256px"
                            ),
                        },
                    },
                    "required": ["image_url"],
                },
            ),
            Tool(
                name="generate_speech",
                description="Generate speech from text using MiniMax Speech-02 HD",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to convert to speech (max 5000 characters)",
                        },
                        "voice_id": {
                            "type": "string",
                            "description": "Voice to use for speech generation",
                            "default": "Wise_Woman",
                        },
                        "speed": {
                            "type": "number",
                            "description": "Speech speed (0.5-2.0)",
                            "minimum": 0.5,
                            "maximum": 2.0,
                            "default": 1.0,
                        },
                        "volume": {
                            "type": "number",
                            "description": "Volume level (0-10)",
                            "minimum": 0,
                            "maximum": 10,
                            "default": 1.0,
                        },
                        "pitch": {
                            "type": "integer",
                            "description": "Voice pitch (-12 to 12)",
                            "minimum": -12,
                            "maximum": 12,
                            "default": 0,
                        },
                        "emotion": {
                            "type": "string",
                            "description": "Emotion of the generated speech",
                            "enum": [
                                "happy",
                                "sad",
                                "angry",
                                "fearful",
                                "disgusted",
                                "surprised",
                                "neutral",
                            ],
                        },
                        "language_boost": {
                            "type": "string",
                            "description": "Enhance recognition of specified languages",
                            "enum": [
                                "Chinese",
                                "English",
                                "Arabic",
                                "Russian",
                                "Spanish",
                                "French",
                                "Portuguese",
                                "German",
                                "auto",
                            ],
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Format of the output",
                            "enum": ["url", "hex"],
                            "default": "url",
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="generate_video",
                description="Generate videos from text or images",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text description of the video to generate",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use for video generation",
                            "enum": [
                                "fal-ai/runway-gen3/turbo/image-to-video",
                                "fal-ai/kling-video/v1/standard/image-to-video",
                                "fal-ai/minimax/video-01",
                            ],
                            "default": "fal-ai/runway-gen3/turbo/image-to-video",
                        },
                        "image_url": {
                            "type": "string",
                            "description": "URL of image to use as first frame (for image-to-video)",
                        },
                        "duration": {
                            "type": "integer",
                            "description": "Duration in seconds",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio of the video",
                            "enum": ["16:9", "9:16", "1:1"],
                            "default": "16:9",
                        },
                    },
                    "required": ["prompt"],
                },
            ),
            Tool(
                name="transcribe_audio",
                description="Convert speech to text using Whisper",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_url": {
                            "type": "string",
                            "description": "URL of the audio file to transcribe",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task to perform",
                            "enum": ["transcribe", "translate"],
                            "default": "transcribe",
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Language of the audio (optional, auto-detected if not specified)"
                            ),
                        },
                        "chunk_level": {
                            "type": "string",
                            "description": "Chunking level for output",
                            "enum": ["segment", "word"],
                            "default": "segment",
                        },
                        "version": {
                            "type": "string",
                            "description": "Whisper model version",
                            "enum": ["3", "3-turbo"],
                            "default": "3",
                        },
                    },
                    "required": ["audio_url"],
                },
            ),
            Tool(
                name="upload_file",
                description="Upload a file to FAL.ai CDN",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Local path to the file to upload",
                        },
                        "content_type": {
                            "type": "string",
                            "description": "MIME type of the file (auto-detected if not provided)",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="check_queue_status",
                description="Check the status of a queued request",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "request_url": {
                            "type": "string",
                            "description": "The request URL returned from a previous operation",
                        },
                    },
                    "required": ["request_url"],
                },
            ),
            Tool(
                name="get_queue_result",
                description="Get the result of a completed queued request",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "request_url": {
                            "type": "string",
                            "description": "The request URL returned from a previous operation",
                        },
                    },
                    "required": ["request_url"],
                },
            ),
        ]

    async def vectorize_image(self, image_url: str) -> Dict[str, Any]:
        """Convert a raster image to SVG format using Recraft."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{FAL_QUEUE_BASE}/fal-ai/recraft/vectorize",
                    headers=self.headers,
                    json={"image_url": image_url},
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise Exception(f"Image vectorization failed: {str(e)}")

    async def generate_speech(self, text: str, **params) -> Dict[str, Any]:
        """Generate speech from text using MiniMax Speech-02 HD."""
        try:
            # Build voice settings
            voice_setting = {
                "voice_id": params.get("voice_id", "Wise_Woman"),
                "speed": params.get("speed", 1.0),
                "vol": params.get("volume", 1.0),
                "pitch": params.get("pitch", 0),
                "english_normalization": False,
            }

            # Add emotion if specified
            if "emotion" in params:
                voice_setting["emotion"] = params["emotion"]

            # Build request parameters
            request_params = {
                "text": text,
                "voice_setting": voice_setting,
                "output_format": params.get("output_format", "url"),
            }

            # Add language boost if specified
            if "language_boost" in params:
                request_params["language_boost"] = params["language_boost"]

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{FAL_QUEUE_BASE}/fal-ai/minimax/speech-02-hd",
                    headers=self.headers,
                    json=request_params,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise Exception(f"Speech generation failed: {str(e)}")

    async def list_models(self, category: str = "all") -> Dict[str, Any]:
        """List available FAL.ai models by category."""
        # Since FAL.ai doesn't have a public models API, we'll return our curated list
        models = {
            "image": [
                {
                    "id": "fal-ai/flux/schnell",
                    "name": "FLUX Schnell",
                    "description": "Fast image generation with FLUX",
                    "category": "image",
                },
                {
                    "id": "fal-ai/flux/dev",
                    "name": "FLUX Dev",
                    "description": "High-quality image generation with FLUX",
                    "category": "image",
                },
                {
                    "id": "fal-ai/flux-pro/v1.1-ultra",
                    "name": "FLUX Pro Ultra",
                    "description": "Ultra high-quality FLUX model",
                    "category": "image",
                },
                {
                    "id": "fal-ai/flux-pro/kontext/text-to-image",
                    "name": "FLUX.1 Kontext [pro]",
                    "description": "State-of-the-art image generation with unprecedented prompt following, photorealistic rendering, and flawless typography",
                    "category": "image",
                },
                {
                    "id": "fal-ai/flux-pro/kontext/max",
                    "name": "FLUX.1 Kontext [max]",
                    "description": "Frontier image editing model with improved prompt adherence and typography generation for editing without compromise on speed",
                    "category": "image",
                },
                {
                    "id": "fal-ai/recraft/v3/text-to-image",
                    "name": "Recraft v3",
                    "description": "Recraft text-to-image model",
                    "category": "image",
                },
                {
                    "id": "fal-ai/hidream-i1-full",
                    "name": "HiDream I1 Full",
                    "description": "HiDream image generation model",
                    "category": "image",
                },
                {
                    "id": "fal-ai/imagen4/preview",
                    "name": "Imagen 4 Preview",
                    "description": "Imagen 4 image generation model",
                    "category": "image",
                },
                {
                    "id": "fal-ai/imagen4/preview/ultra",
                    "name": "Imagen 4 Ultra",
                    "description": "Google's highest quality image generation model",
                    "category": "image",
                },
            ],
            "video": [
                {
                    "id": "fal-ai/runway-gen3/turbo/image-to-video",
                    "name": "Runway Gen3 Turbo (Image-to-Video)",
                    "description": "Convert images to videos with Runway Gen3",
                    "category": "video",
                },
                {
                    "id": "fal-ai/runway-gen3/turbo/text-to-video",
                    "name": "Runway Gen3 Turbo (Text-to-Video)",
                    "description": "Generate videos from text with Runway Gen3",
                    "category": "video",
                },
                {
                    "id": "fal-ai/kling-video/v2/master/image-to-video",
                    "name": "Kling Video v2 (Image-to-Video)",
                    "description": "Convert images to videos with Kling",
                    "category": "video",
                },
                {
                    "id": "fal-ai/kling-video/v2/master/text-to-video",
                    "name": "Kling Video v2 (Text-to-Video)",
                    "description": "Generate videos from text with Kling",
                    "category": "video",
                },
                {
                    "id": "fal-ai/minimax/video-01/image-to-video",
                    "name": "MiniMax Video-01 (Image-to-Video)",
                    "description": "Convert images to videos with MiniMax",
                    "category": "video",
                },
            ],
            "audio": [
                {
                    "id": "fal-ai/whisper",
                    "name": "Whisper",
                    "description": "Speech-to-text transcription",
                    "category": "audio",
                },
                {
                    "id": "fal-ai/minimax/speech-02-hd",
                    "name": "MiniMax Speech-02 HD",
                    "description": "High-quality text-to-speech",
                    "category": "audio",
                },
            ],
        }

        if category == "all":
            all_models = []
            for cat_models in models.values():
                all_models.extend(cat_models)
            return {"models": all_models, "total": len(all_models)}
        elif category in models:
            return {"models": models[category], "total": len(models[category])}
        else:
            return {"models": [], "total": 0}

    async def search_models(self, query: str) -> Dict[str, Any]:
        """Search for models by keywords."""
        all_models_response = await self.list_models("all")
        all_models = all_models_response["models"]

        query_lower = query.lower()
        matching_models = []

        for model in all_models:
            if (
                query_lower in model["name"].lower()
                or query_lower in model["description"].lower()
                or query_lower in model["id"].lower()
            ):
                matching_models.append(model)

        return {
            "models": matching_models,
            "total": len(matching_models),
            "query": query,
        }

    async def get_model_schema(self, model: str) -> Dict[str, Any]:
        """Get the input schema for a specific model."""
        # Return schema information for known models
        schemas = {
            "fal-ai/flux/schnell": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt",
                    },
                    "image_size": {
                        "type": "string",
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
                        "default": 4,
                        "min": 1,
                        "max": 50,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                    },
                    "num_images": {"type": "integer", "default": 1, "min": 1, "max": 4},
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                },
            },
            "fal-ai/flux/dev": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt",
                    },
                    "image_size": {
                        "type": "string",
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
                        "default": 28,
                        "min": 1,
                        "max": 50,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                    },
                    "num_images": {"type": "integer", "default": 1, "min": 1, "max": 4},
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                },
            },
            "fal-ai/flux-pro/kontext/text-to-image": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "default": "1:1",
                        "enum": [
                            "21:9",
                            "16:9",
                            "4:3",
                            "3:2",
                            "1:1",
                            "2:3",
                            "3:4",
                            "9:16",
                            "9:21",
                        ],
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 3.5,
                        "description": "CFG scale for prompt adherence",
                    },
                    "num_images": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of images to generate",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                    "safety_tolerance": {
                        "type": "string",
                        "default": "2",
                        "enum": ["1", "2", "3", "4", "5", "6"],
                        "description": "Safety tolerance level (1=strict, 6=permissive)",
                    },
                    "output_format": {
                        "type": "string",
                        "default": "jpeg",
                        "enum": ["jpeg", "png"],
                        "description": "Output image format",
                    },
                    "sync_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "Wait for generation to complete before returning",
                    },
                },
            },
            "fal-ai/flux-pro/kontext/max": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt for editing the image",
                    },
                    "image_url": {
                        "type": "string",
                        "required": True,
                        "description": "URL of the input image to edit",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": [
                            "21:9",
                            "16:9",
                            "4:3",
                            "3:2",
                            "1:1",
                            "2:3",
                            "3:4",
                            "9:16",
                            "9:21",
                        ],
                        "description": "Aspect ratio of the output image",
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 3.5,
                        "description": "CFG scale for prompt adherence",
                    },
                    "num_images": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of images to generate",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                    "safety_tolerance": {
                        "type": "string",
                        "default": "2",
                        "enum": ["1", "2", "3", "4", "5", "6"],
                        "description": "Safety tolerance level (1=strict, 6=permissive)",
                    },
                    "output_format": {
                        "type": "string",
                        "default": "jpeg",
                        "enum": ["jpeg", "png"],
                        "description": "Output image format",
                    },
                    "sync_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "Wait for generation to complete before returning",
                    },
                },
            },
            "fal-ai/whisper": {
                "model": model,
                "parameters": {
                    "audio_url": {
                        "type": "string",
                        "required": True,
                        "description": "URL of audio file",
                    },
                    "task": {
                        "type": "string",
                        "default": "transcribe",
                        "enum": ["transcribe", "translate"],
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (auto-detected if not specified)",
                    },
                    "chunk_level": {
                        "type": "string",
                        "default": "segment",
                        "enum": ["segment", "word"],
                    },
                    "version": {
                        "type": "string",
                        "default": "3",
                        "enum": ["3", "3-turbo"],
                    },
                },
            },
            "fal-ai/imagen4/preview": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt",
                    },
                    "image_size": {
                        "type": "string",
                        "default": "landscape_4_3",
                        "enum": [
                            "square_hd",
                            "square",
                            "portrait_4_3",
                            "portrait_16_9",
                            "landscape_4_3",
                            "landscape_16_9",
                        ],
                        "description": "Size of the generated image",
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "default": 28,
                        "min": 1,
                        "max": 50,
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 3.5,
                        "description": "Guidance scale for generation",
                    },
                    "num_images": {
                        "type": "integer",
                        "default": 1,
                        "min": 1,
                        "max": 4,
                    },
                },
            },
            "fal-ai/imagen4/preview/ultra": {
                "model": model,
                "parameters": {
                    "prompt": {
                        "type": "string",
                        "required": True,
                        "description": "Text prompt",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "default": "1:1",
                        "enum": ["1:1", "16:9", "9:16", "3:4", "4:3"],
                        "description": "Aspect ratio of the generated image",
                    },
                    "num_images": {
                        "type": "integer",
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "description": "Number of images to generate",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducible results",
                    },
                },
            },
        }

        if model in schemas:
            return schemas[model]
        else:
            return {
                "model": model,
                "error": f"Schema not available for model: {model}",
                "suggestion": "Use list_models to see available models",
            }


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
                            "fal-ai/flux-pro/kontext/text-to-image",
                            "fal-ai/flux-pro/kontext/max",
                            "fal-ai/recraft/v3/text-to-image",
                            "fal-ai/hidream-i1-full",
                            "fal-ai/imagen4/preview",
                            "fal-ai/imagen4/preview/ultra",
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
        Tool(
            name="list_models",
            description="List available FAL.ai models by category",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Model category to filter by",
                        "enum": ["image", "video", "audio", "all"],
                        "default": "all",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="search_models",
            description="Search for FAL.ai models by keywords",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for model names or descriptions",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_model_schema",
            description="Get the input schema and parameters for a specific FAL.ai model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model identifier (e.g., 'fal-ai/flux/schnell')",
                    }
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="vectorize_image",
            description="Convert raster images (PNG, JPG, WEBP) to SVG vector format using Recraft",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "URL of the image to vectorize. Must be PNG, JPG or WEBP, less than 5MB, resolution less than 16MP, max dimension less than 4096px, min dimension more than 256px",
                    }
                },
                "required": ["image_url"],
            },
        ),
        Tool(
            name="generate_speech",
            description="Generate speech from text using MiniMax Speech-02 HD with voice customization",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech (max 5000 characters)",
                    },
                    "voice_id": {
                        "type": "string",
                        "description": "Voice to use for speech generation",
                        "default": "Wise_Woman",
                    },
                    "speed": {
                        "type": "number",
                        "description": "Speech speed (0.5-2.0)",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "default": 1.0,
                    },
                    "volume": {
                        "type": "number",
                        "description": "Volume level (0-10)",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 1.0,
                    },
                    "pitch": {
                        "type": "integer",
                        "description": "Voice pitch (-12 to 12)",
                        "minimum": -12,
                        "maximum": 12,
                        "default": 0,
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Emotion of the generated speech",
                        "enum": [
                            "happy",
                            "sad",
                            "angry",
                            "fearful",
                            "disgusted",
                            "surprised",
                            "neutral",
                        ],
                    },
                    "language_boost": {
                        "type": "string",
                        "description": "Enhance recognition of specified languages",
                        "enum": [
                            "Chinese",
                            "English",
                            "Arabic",
                            "Russian",
                            "Spanish",
                            "French",
                            "Portuguese",
                            "German",
                            "auto",
                        ],
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Format of the output",
                        "enum": ["url", "hex"],
                        "default": "url",
                    },
                },
                "required": ["text"],
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

        elif name == "list_models":
            category = arguments.get("category", "all")
            result = await fal_client.list_models(category)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "search_models":
            query = arguments["query"]
            result = await fal_client.search_models(query)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_model_schema":
            model = arguments["model"]
            result = await fal_client.get_model_schema(model)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "vectorize_image":
            image_url = arguments["image_url"]
            result = await fal_client.vectorize_image(image_url)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "generate_speech":
            text = arguments["text"]
            # Remove text from arguments to avoid duplicate parameter
            speech_params = {k: v for k, v in arguments.items() if k != "text"}
            result = await fal_client.generate_speech(text, **speech_params)
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
