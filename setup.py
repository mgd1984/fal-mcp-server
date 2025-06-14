#!/usr/bin/env python3
"""
Setup script for FAL.ai MCP Server
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="fal-ai-mcp-server",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Model Context Protocol server for FAL.ai generative AI platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fal-mcp",
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fal-mcp=main:main",
        ],
    },
    keywords="mcp model-context-protocol fal.ai ai generative-ai image-generation video-generation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/fal-mcp/issues",
        "Source": "https://github.com/yourusername/fal-mcp",
        "Documentation": "https://github.com/yourusername/fal-mcp#readme",
    },
)
