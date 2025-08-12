# mcp_server.py

import sys
import json
from typing import List, Union
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from find_similar_assets.image_similarity import find_similar_assets

# Create FastMCP app
mcp = FastMCP()

# Define tool schemas
class FindSimilarImagesInput(BaseModel):
    """Find images similar to the input image using advanced neural network embeddings."""
    image_path: str = Field(
        ..., 
        description="Absolute path to the query image that will be used as the reference for finding similar images"
    )
    assets_directory: str = Field(
        ..., 
        description="Directory path containing potential matching images to search through (supports PNG and JPG files)"
    )
    top_k: int = Field(
        3, 
        description="Maximum number of similar images to return, sorted by similarity score in descending order"
    )

class ImageMatch(BaseModel):
    """Information about a matching image with similarity metrics."""
    file: str = Field(..., description="Filename of the matching image")
    path: str = Field(..., description="Full path to the matching image")
    similarity: float = Field(..., description="Similarity score between 0 and 1, where 1 indicates perfect similarity (identical images) and values closer to 0 indicate less similarity")

class FindSimilarImagesOutput(BaseModel):
    """Results of the image similarity search sorted by visual similarity."""
    matches: List[ImageMatch] = Field(..., description="List of matching images sorted by similarity score in descending order")

# Register the tool with FastMCP
@mcp.tool(
    name="find_similar_images",
    description="""Find images that are visually similar to a query image using CLIP neural network embeddings. This tool analyzes visual features to identify images with similar content, objects, or style - not just metadata.

**Key use cases include:**
• Helping developers check if an image shared with an LLM already exists in the workspace assets
• Finding duplicate or similar images across project directories
• Organizing visual assets by similarity
• Validating that new images don't duplicate existing ones before adding them to a project

**Required inputs:**
• input_data: Either a FindSimilarImagesInput object or a JSON string containing the required parameters
  - image_path: When an image is shared with an LLM, it can be temporarily stored to obtain this absolute path
  - assets_directory: The directory containing potential matching images to search through
  - top_k: Maximum number of similar images to return""",
)
async def find_similar_images(input_data: Union[FindSimilarImagesInput, str]) -> FindSimilarImagesOutput:
    """Find images similar to the input image."""
    # Handle both structured input and JSON string input
    if isinstance(input_data, str):
        try:
            # Parse JSON string and create FindSimilarImagesInput object
            json_data = json.loads(input_data)
            parsed_input = FindSimilarImagesInput(**json_data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid JSON string provided: {e}")
    else:
        parsed_input = input_data
    
    results = find_similar_assets(
        parsed_input.image_path, 
        parsed_input.assets_directory, 
        parsed_input.top_k
    )
    
    return FindSimilarImagesOutput(
        matches=[
            ImageMatch(
                file=match["name"],
                path=match["path"],
                similarity=match["similarity"]
            )
            for match in results
        ]
    )

if __name__ == "__main__":
    # Run the MCP server via stdio
    mcp.run(transport="stdio")
