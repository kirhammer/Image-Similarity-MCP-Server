# mcp_server.py

import sys
from typing import List
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from find_similar_assets.image_similarity import find_similar_assets

# Create FastMCP app
mcp = FastMCP()

# Define tool schemas
class FindSimilarImagesInput(BaseModel):
    """Find images similar to the input image."""
    image_path: str = Field(..., description="Path to the query image")
    assets_directory: str = Field(..., description="Directory containing potential matching images")
    top_k: int = Field(3, description="Number of similar images to return")

class ImageMatch(BaseModel):
    """Information about a matching image."""
    file: str = Field(..., description="Filename of the matching image")
    path: str = Field(..., description="Full path to the matching image")
    similarity: float = Field(..., description="Similarity score (0-1)")

class FindSimilarImagesOutput(BaseModel):
    """Results of the image similarity search."""
    matches: List[ImageMatch] = Field(..., description="List of matching images")

# Register the tool with FastMCP
@mcp.tool(
    name="find_similar_images",
    description="Find images that are visually similar to a query image"
)
async def find_similar_images(input_data: FindSimilarImagesInput) -> FindSimilarImagesOutput:
    """Find images similar to the input image."""
    results = find_similar_assets(
        input_data.image_path, 
        input_data.assets_directory, 
        input_data.top_k
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
