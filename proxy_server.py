from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.logger import logger as fastapi_logger
import httpx
import asyncio
import json
from typing import AsyncGenerator, Dict, Any

# Use FastAPI's logger
logger = fastapi_logger

app = FastAPI(
    debug=True,  # Enable debug mode for more detailed logging
    title="Tool Proxy",
    description="A proxy server that adds tool definitions to LLM requests"
)

class ToolProxy:
    def __init__(self, target_url: str):
        self.target_url = target_url
        self.tools = self._define_tools()
        logger.info(f"Initialized ToolProxy with target URL: {target_url}")
    
    def _define_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "example_tool",
                    "description": "An example tool",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        logger.debug(f"Defined tools: {json.dumps(tools)}")
        return tools
    
    async def modify_request(self, request_data: dict) -> dict:
        logger.debug(f"Original request: {json.dumps(request_data)}")
        if "tools" in request_data:
            logger.info("Extending existing tools in request")
            request_data["tools"].extend(self.tools)
        else:
            logger.info("Adding tools to request")
            request_data["tools"] = self.tools
        logger.debug(f"Modified request: {json.dumps(request_data)}")
        return request_data
    
    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
        result = {"result": "example result"}
        logger.debug(f"Tool execution result: {result}")
        return result

    async def handle_tool_calls(self, response_data: dict) -> dict:
        if "tool_calls" not in response_data:
            logger.debug("No tool calls in response")
            return response_data
            
        logger.info(f"Processing {len(response_data['tool_calls'])} tool calls")
        tool_results = []
        for tool_call in response_data["tool_calls"]:
            result = await self.execute_tool(tool_call)
            tool_results.append(result)
            
        response_data["tool_results"] = tool_results
        logger.debug(f"Response with tool results: {json.dumps(response_data)}")
        return response_data

    async def process_chunk(self, chunk: str) -> str:
        """Process a single SSE chunk."""
        if not chunk.strip() or chunk.startswith(":"):
            return chunk

        logger.debug(f"Processing chunk: {chunk}")
        try:
            data = json.loads(chunk.removeprefix("data: "))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse chunk as JSON: {chunk}")
            return chunk

        if "tool_calls" in data:
            logger.info("Found tool calls in streaming chunk")
            data = await self.handle_tool_calls(data)

        processed_chunk = f"data: {json.dumps(data)}\n\n"
        logger.debug(f"Processed chunk: {processed_chunk}")
        return processed_chunk

    async def stream_processor(self, stream: AsyncGenerator) -> AsyncGenerator:
        """Process a stream of SSE events."""
        logger.info("Starting stream processing")
        buffer = ""
        
        try:
            async for chunk in stream:
                chunk = chunk.decode('utf-8')
                buffer += chunk
                logger.debug(f"Current buffer size: {len(buffer)} bytes")
                
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    processed_chunk = await self.process_chunk(message)
                    if processed_chunk:
                        yield processed_chunk.encode('utf-8')
            
            if buffer:
                logger.debug("Processing remaining buffer")
                processed_chunk = await self.process_chunk(buffer)
                if processed_chunk:
                    yield processed_chunk.encode('utf-8')
        except Exception as e:
            logger.error(f"Error during stream processing: {str(e)}", exc_info=True)
            raise

        logger.info("Stream processing completed")

@app.post("/v1/chat/completions")
async def proxy_request(request: Request):
    proxy = ToolProxy("http://127.0.0.1:52415")
    
    try:
        request_data = await request.json()
        logger.info("Received chat completion request")
        logger.debug(f"Request data: {json.dumps(request_data)}")
        
        is_streaming = request_data.get("stream", False)
        logger.info(f"Streaming mode: {is_streaming}")
        
        modified_request = await proxy.modify_request(request_data)
        
        async with httpx.AsyncClient() as client:
            logger.info(f"Forwarding request to: {proxy.target_url}")
            response = await client.post(
                f"{proxy.target_url}/v1/chat/completions",
                json=modified_request,
                headers={"Accept": "text/event-stream"} if is_streaming else {}
            )
            
            if response.status_code != 200:
                logger.error(f"Upstream server returned status code: {response.status_code}")
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            
            if is_streaming:
                logger.info("Returning streaming response")
                return StreamingResponse(
                    proxy.stream_processor(response.aiter_bytes()),
                    media_type="text/event-stream"
                )
            else:
                logger.info("Processing non-streaming response")
                response_data = response.json()
                modified_response = await proxy.handle_tool_calls(response_data)
                return modified_response
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise 