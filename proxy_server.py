from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    debug=True,  # Enable debug mode for more detailed logging
    title="Tool Proxy",
    description="A proxy server that adds tool definitions to LLM requests"
)


@app.post("/v1/chat/completions")
async def proxy_request(request: Request) -> Response:
    # Get the raw request body
    body = await request.json()
    logger.info(f"Incoming request: {json.dumps(body, indent=2)}")
    
    # Check if streaming is requested
    stream = body.get('stream', False)
    
    if not stream:
        async with httpx.AsyncClient() as client:
            ollama_response = await client.post(
                "http://localhost:11434/v1/chat/completions",
                json=body
            )
            response_body = ollama_response.json()
            logger.info(f"Ollama response: {json.dumps(response_body, indent=2)}")
            return Response(
                content=ollama_response.content,
                status_code=ollama_response.status_code,
                headers=dict(ollama_response.headers)
            )
    
    # Handle streaming response
    async def stream_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/v1/chat/completions",
                json=body
            ) as response:
                async for chunk in response.aiter_bytes():
                    logger.info(f"Streaming chunk: {chunk.decode()}")
                    yield chunk
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
    
