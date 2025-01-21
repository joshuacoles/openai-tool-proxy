from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
import logging
import asyncio
from typing import Optional, Dict, Any

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define test tool
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform basic arithmetic calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

# Define our tools to inject
PROXY_TOOLS = [CALCULATOR_TOOL]

app = FastAPI(
    debug=True,  # Enable debug mode for more detailed logging
    title="Tool Proxy",
    description="A proxy server that adds tool definitions to LLM requests"
)


class StreamParser:
    def __init__(self):
        self.buffer = ""
        self.message_count = 0
    
    def parse_chunk(self, chunk: bytes) -> Optional[Dict[str, Any]]:
        """Parse a chunk of bytes into a JSON object."""
        try:
            text = chunk.decode('utf-8')
            logger.debug(f"Raw chunk text: {text}")
            
            # Handle server-sent events format
            if text.startswith('data: '):
                text = text.removeprefix('data: ')
            
            # Skip empty chunks
            if not text.strip():
                logger.debug("Skipping empty chunk")
                return None
            
            # Try to parse as JSON
            try:
                data = json.loads(text)
                self.message_count += 1
                logger.debug(f"Successfully parsed message {self.message_count}: {json.dumps(data, indent=2)}")
                
                # Check for tool calls
                if 'tool_calls' in data:
                    logger.info(f"Found tool calls in message: {json.dumps(data['tool_calls'], indent=2)}")
                
                return data
            
            except json.JSONDecodeError:
                # Add to buffer and try to parse
                self.buffer += text
                logger.debug(f"Added to buffer. Current buffer: {self.buffer}")
                
                try:
                    data = json.loads(self.buffer)
                    self.buffer = ""  # Clear buffer on successful parse
                    self.message_count += 1
                    logger.debug(f"Successfully parsed buffered message {self.message_count}: {json.dumps(data, indent=2)}")
                    
                    # Check for tool calls
                    if 'tool_calls' in data:
                        logger.info(f"Found tool calls in buffered message: {json.dumps(data['tool_calls'], indent=2)}")
                    
                    return data
                
                except json.JSONDecodeError:
                    logger.debug("Incomplete JSON in buffer, waiting for more chunks")
                    return None
        
        except Exception as e:
            logger.error(f"Error parsing chunk: {str(e)}")
            return None


@app.post("/v1/chat/completions")
async def proxy_request(request: Request) -> Response:
    logger.info("Received chat completion request")
    
    # Get the raw request body
    body = await request.json()
    logger.info(f"Incoming request: {json.dumps(body, indent=2)}")
    
    # Add tools to the request, preserving any existing tools
    body["tools"] = body.get("tools", []) + PROXY_TOOLS
    logger.info(f"Modified request with tools: {json.dumps(body, indent=2)}")
    
    # Check if streaming is requested
    stream = body.get('stream', False)
    logger.info(f"Stream mode: {stream}")
    
    timeout = httpx.Timeout(10.0, connect=10.0)
    
    if not stream:
        logger.info("Handling non-streaming request")
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Sending request to Ollama")
            ollama_response = await client.post(
                "http://localhost:11434/v1/chat/completions",
                json=body
            )
            response_body = ollama_response.json()
            logger.info(f"Received response from Ollama: {json.dumps(response_body, indent=2)}")
            return Response(
                content=ollama_response.content,
                status_code=ollama_response.status_code,
                headers=dict(ollama_response.headers)
            )
    
    # Handle streaming response
    logger.info("Handling streaming request")
    parser = StreamParser()
    
    async def stream_response():
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Starting streaming request to Ollama")
            async with client.stream(
                "POST",
                "http://localhost:11434/v1/chat/completions",
                json=body
            ) as response:
                logger.info("Established streaming connection with Ollama")
                async for chunk in response.aiter_bytes():
                    parsed = parser.parse_chunk(chunk)
                    if parsed:
                        # Re-encode as SSE format
                        yield f"data: {json.dumps(parsed)}\n\n".encode('utf-8')
                    else:
                        # Forward unparseable chunks as-is
                        yield chunk
    
    logger.info("Returning streaming response")
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
    
