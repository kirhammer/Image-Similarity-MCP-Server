# Image Finder MCP Server

This is an image finder tool that uses CLIP (Contrastive Language-Image Pre-Training) to find visually similar images based on a query image. It's implemented as a Model Context Protocol (MCP) server that can be used with compatible clients, as well as a standalone FastAPI server.

## Features

- Find visually similar images based on a query image using CLIP embeddings
- Configurable number of matches to return
- Caches image embeddings for improved performance
- Works with PNG and JPEG image formats
- Provides similarity scores for each match

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd image-finder
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

The project requires the following main dependencies:
- torch & torchvision: For tensor operations and image processing
- transformers: For the CLIP model
- pillow: For image handling
- fastapi & uvicorn: For the API server
- fastmcp: For the MCP server implementation

## Usage

### As an MCP Server (stdio)

The image finder can be run as an MCP server using stdio transport. This allows it to be used with any MCP client that supports stdio communication, such as VS Code with the MCP extension.

1. Make sure your virtual environment is activated
2. Run the server:
   ```bash
   python mcp_server.py
   ```

The MCP server communicates over stdio and exposes the `find_similar_images` tool.

### As a FastAPI Server (HTTP)

The tool can also be run as a standard HTTP API server using FastAPI:

1. Make sure your virtual environment is activated
2. Run the server:
   ```bash
   uvicorn server:app --reload
   ```

This will start the server at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### Client Configuration

To use this tool with an MCP client (like VS Code with the MCP extension), you need to configure the client to connect to this server.

#### VS Code Configuration

1. Install the MCP extension for VS Code if you haven't already

2. Create a `.vscode/mcp.json` file in your project with the following content:

```json
{
    "servers": {
        "Image Finder": {
            "type": "stdio",
            "command": "/path/to/project/image-finder/.venv/bin/python",
            "args": [
                "/path/to/project/image-finder/mcp_server.py"
            ],
            "cwd": "/path/to/project/image-finder"
        }
    }
}
```

3. **Important**: The paths above are specific to a local setup and won't work on your system. You must replace them with your own local paths:
   - Replace all instances of `/path/to/project/image-finder/` with the absolute path to your project directory
   - Use the full path to your Python interpreter in the virtual environment
   - Use the full path to the `mcp_server.py` script
   - Specify the `cwd` (current working directory) parameter as your project's root directory

4. Restart VS Code or reload the window

## Tool API

The tool provides the following API:

### MCP Tool: `find_similar_images`

Finds images that are visually similar to a query image.

**Input:**
- `image_path`: Path to the query image
- `assets_directory`: Directory containing potential matching images
- `top_k`: Number of similar images to return (default: 3)

**Output:**
- `matches`: List of matching images with the following properties:
  - `file`: Filename of the matching image
  - `path`: Full path to the matching image
  - `similarity`: Similarity score (0-1)

### HTTP API Endpoint: `POST /findSimilarAssets`

**Request Body:**
```json
{
  "imagePath": "path/to/query/image.jpg",
  "assetsDirectory": "path/to/assets/directory",
  "topK": 3
}
```

**Response:**
```json
{
  "matches": [
    {
      "file": "matching_image1.jpg",
      "path": "/full/path/to/matching_image1.jpg",
      "similarity": 0.95
    },
    {
      "file": "matching_image2.jpg",
      "path": "/full/path/to/matching_image2.jpg",
      "similarity": 0.85
    }
  ]
}
```

## Example

### Using with VS Code Copilot

When configured with VS Code Copilot, you can use the tool like this:

```
Find images similar to /full/path/to/query.jpg
```

**Important**: You must include the full absolute path to the query image in your prompt. The assets directory will be automatically determined by the agent. For example:

```
Find images similar to /Users/username/personal/image-finder/assets/query.jpg
```

The tool will return a list of similar images with their similarity scores.

<img width="424" alt="Captura de pantalla 2025-06-25 a la(s) 9 00 23 a m" src="https://github.com/user-attachments/assets/472dc2f1-d6b4-4fd0-8864-1c29cc494ebe" />

### Using with curl (for HTTP API)

```bash
curl -X POST http://localhost:8000/findSimilarAssets \
  -H "Content-Type: application/json" \
  -d '{
    "imagePath": "assets/query.jpg",
    "assetsDirectory": "assets",
    "topK": 5
  }'
```

## How It Works

The image finder uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model to generate embeddings for images. These embeddings capture the visual features of the images in a high-dimensional space. The similarity between images is calculated using cosine similarity between their embeddings.

For efficiency, the tool caches the embeddings of previously processed images in a pickle file (`embeddings_cache.pkl`). This significantly speeds up subsequent searches involving the same images.

## Project Structure

- `mcp_server.py`: MCP server implementation using stdio transport
- `server.py`: FastAPI server implementation for HTTP access
- `find_similar_assets/`: Module containing the image similarity functionality
  - `image_similarity.py`: Core implementation using CLIP for image embeddings
  - `embeddings_cache.pkl`: Cache file for storing image embeddings
- `assets/`: Directory to store image assets (place your images here)
- `temp/`: Temporary directory for processing
- `requirements.txt`: Python dependencies
- `.vscode/mcp.json`: MCP server configuration for VS Code

## Dependencies

Main dependencies include:
- torch & torchvision
- transformers (for CLIP model)
- PIL (Pillow)
- fastapi & uvicorn
- fastmcp

See `requirements.txt` for the complete list.

## Tips

1. For best performance, place your images in the `assets` directory
2. The first run will download the CLIP model and be slower, subsequent runs will be faster
3. Larger images take more time to process
4. Similarity scores range from 0 to 1, with 1 being identical images

## License

[Your license information here]

## Acknowledgements

This project uses OpenAI's CLIP model for image embedding generation.
