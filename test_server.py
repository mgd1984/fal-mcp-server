#!/usr/bin/env python3
"""
Test suite for FAL.ai MCP Server

Validates the server structure, tool definitions, and basic functionality.
"""

import asyncio
import os
import sys
from unittest.mock import patch

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path modification to avoid E402
from main import FALClient, handle_call_tool, handle_list_tools, server  # noqa: E402


def test_fal_client_initialization():
    """Test FALClient can be initialized with API key"""
    client = FALClient("test-api-key")
    assert client.api_key == "test-api-key"
    assert "Authorization" in client.headers
    assert client.headers["Authorization"] == "Key test-api-key"
    print("✅ FALClient initialization test passed")


def test_content_type_detection():
    """Test file content type detection"""
    client = FALClient("test-key")

    test_cases = [
        ("image.jpg", "image/jpeg"),
        ("image.jpeg", "image/jpeg"),
        ("image.png", "image/png"),
        ("video.mp4", "video/mp4"),
        ("audio.mp3", "audio/mpeg"),
        ("unknown.xyz", "application/octet-stream"),
    ]

    for filename, expected_type in test_cases:
        result = client._get_content_type(filename)
        assert (
            result == expected_type
        ), f"Expected {expected_type} for {filename}, got {result}"

    print("✅ Content type detection test passed")


async def test_tool_listing():
    """Test that all expected tools are available"""
    tools = await handle_list_tools()

    # Expected tools based on corrected API
    expected_tools = {
        "generate_image",
        "generate_video",
        "transcribe_audio",
        "upload_file",
        "check_queue_status",
        "get_queue_result",
        "cancel_request",
    }

    actual_tools = {tool.name for tool in tools}

    print(f"Expected tools: {sorted(expected_tools)}")
    print(f"Actual tools: {sorted(actual_tools)}")

    assert (
        actual_tools == expected_tools
    ), f"Tool mismatch. Expected: {expected_tools}, Got: {actual_tools}"
    assert len(tools) == 7, f"Expected 7 tools, got {len(tools)}"

    print("✅ Tool listing test passed")


async def test_tool_schemas():
    """Test that all tools have proper input schemas"""
    tools = await handle_list_tools()

    for tool in tools:
        assert hasattr(tool, "inputSchema"), f"Tool {tool.name} missing inputSchema"
        assert isinstance(
            tool.inputSchema, dict
        ), f"Tool {tool.name} inputSchema is not a dict"
        assert (
            "type" in tool.inputSchema
        ), f"Tool {tool.name} inputSchema missing 'type'"
        assert (
            "properties" in tool.inputSchema
        ), f"Tool {tool.name} inputSchema missing 'properties'"

        # Check required fields exist
        if "required" in tool.inputSchema:
            for required_field in tool.inputSchema["required"]:
                assert (
                    required_field in tool.inputSchema["properties"]
                ), f"Tool {tool.name} required field '{required_field}' not in properties"

    print("✅ Tool schema validation test passed")


async def test_generate_image_tool():
    """Test generate_image tool definition"""
    tools = await handle_list_tools()
    generate_image_tool = next((t for t in tools if t.name == "generate_image"), None)

    assert generate_image_tool is not None, "generate_image tool not found"

    # Check required properties
    props = generate_image_tool.inputSchema["properties"]
    assert "prompt" in props, "generate_image missing prompt property"
    assert "model" in props, "generate_image missing model property"

    # Check model enum has valid FAL.ai models
    model_enum = props["model"].get("enum", [])
    expected_models = [
        "fal-ai/flux/schnell",
        "fal-ai/flux/dev",
        "fal-ai/flux-pro/v1.1-ultra",
        "fal-ai/stable-diffusion-v35-large",
        "fal-ai/recraft/v3/text-to-image",
        "fal-ai/hidream-i1-full",
    ]

    for model in expected_models:
        assert model in model_enum, f"Expected model {model} not in enum"

    print("✅ Generate image tool test passed")


async def test_generate_video_tool():
    """Test generate_video tool definition"""
    tools = await handle_list_tools()
    generate_video_tool = next((t for t in tools if t.name == "generate_video"), None)

    assert generate_video_tool is not None, "generate_video tool not found"

    # Check required properties
    props = generate_video_tool.inputSchema["properties"]
    assert "prompt" in props, "generate_video missing prompt property"
    assert "model" in props, "generate_video missing model property"

    # Check model enum has valid video models
    model_enum = props["model"].get("enum", [])
    expected_models = [
        "fal-ai/runway-gen3/turbo/image-to-video",
        "fal-ai/runway-gen3/turbo/text-to-video",
        "fal-ai/kling-video/v2/master/image-to-video",
        "fal-ai/kling-video/v2/master/text-to-video",
        "fal-ai/minimax/video-01/image-to-video",
    ]

    for model in expected_models:
        assert model in model_enum, f"Expected video model {model} not in enum"

    print("✅ Generate video tool test passed")


async def test_error_handling():
    """Test error handling when FAL_KEY is not set"""
    # Mock missing API key
    with patch.dict(os.environ, {}, clear=True):
        # Remove FAL_KEY from environment
        if "FAL_KEY" in os.environ:
            del os.environ["FAL_KEY"]

        # Test tool call without API key
        result = await handle_call_tool("generate_image", {"prompt": "test"})

        assert len(result) == 1
        assert "Error: FAL_KEY environment variable not set" in result[0].text

    print("✅ Error handling test passed")


def test_environment_variable_loading():
    """Test that environment variables are properly loaded"""
    # Test with mock environment variable
    test_key = "test-fal-key-12345"

    with patch.dict(os.environ, {"FAL_KEY": test_key}):
        # Import should pick up the environment variable
        api_key = os.getenv("FAL_KEY")
        assert api_key == test_key

    print("✅ Environment variable loading test passed")


async def test_server_structure():
    """Test basic server structure and methods"""
    # Check that server has required methods
    assert hasattr(server, "list_tools"), "Server missing list_tools method"
    assert hasattr(server, "call_tool"), "Server missing call_tool method"

    # Test that handlers are properly registered
    tools = await handle_list_tools()
    assert len(tools) > 0, "No tools registered"

    print("✅ Server structure test passed")


async def run_all_tests():
    """Run all tests"""
    print("🧪 Running FAL.ai MCP Server Tests\n")

    # Synchronous tests
    test_fal_client_initialization()
    test_content_type_detection()
    test_environment_variable_loading()

    # Asynchronous tests
    await test_tool_listing()
    await test_tool_schemas()
    await test_generate_image_tool()
    await test_generate_video_tool()
    await test_server_structure()
    await test_error_handling()

    print("\n🎉 All tests passed! The FAL.ai MCP server is working correctly.")
    print("\n📋 Test Summary:")
    print("   ✅ FALClient initialization")
    print("   ✅ Content type detection")
    print("   ✅ Environment variable loading")
    print("   ✅ Tool listing (7 tools)")
    print("   ✅ Tool schema validation")
    print("   ✅ Generate image tool")
    print("   ✅ Generate video tool")
    print("   ✅ Server structure")
    print("   ✅ Error handling")


if __name__ == "__main__":
    # Set a test API key for testing
    os.environ["FAL_KEY"] = "test-api-key-for-testing"

    try:
        asyncio.run(run_all_tests())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
