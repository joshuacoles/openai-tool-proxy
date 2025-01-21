from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

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

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

class CalculatorTool:
    @staticmethod
    async def execute(args: Dict[str, Any]) -> Any:
        operation = args['operation']
        a = args['a']
        b = args['b']
        
        logger.info(f"Executing calculator tool: {operation}({a}, {b})")
        
        result = {
            'add': lambda: a + b,
            'subtract': lambda: a - b,
            'multiply': lambda: a * b,
            'divide': lambda: a / b if b != 0 else "Error: Division by zero"
        }[operation]()
        
        logger.info(f"Calculator result: {result}")
        return result

class ToolExecutor:
    def __init__(self):
        self.tools = {
            'calculate': CalculatorTool.execute
        }
        logger.info(f"ToolExecutor initialized with tools: {list(self.tools.keys())}")
    
    async def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return the result in the expected format."""
        logger.info(f"Executing tool call: id={tool_call.id}, name={tool_call.name}, args={tool_call.arguments}")
        
        if tool_call.name not in self.tools:
            logger.error(f"Unknown tool: {tool_call.name}, available tools: {list(self.tools.keys())}")
            return {
                "tool_call_id": tool_call.id,
                "output": f"Error: Unknown tool {tool_call.name}"
            }
        
        try:
            logger.debug(f"Invoking tool {tool_call.name} with arguments: {tool_call.arguments}")
            result = await self.tools[tool_call.name](tool_call.arguments)
            logger.info(f"Tool execution successful: id={tool_call.id}, result={result}")
            return {
                "tool_call_id": tool_call.id,
                "output": str(result)
            }
        except Exception as e:
            logger.error(f"Tool execution error for {tool_call.name}: {str(e)}", exc_info=True)
            return {
                "tool_call_id": tool_call.id,
                "output": f"Error: {str(e)}"
            }

class StreamParser:
    def __init__(self):
        self.buffer = ""
        self.message_count = 0
        self.tool_executor = ToolExecutor()
        logger.info("StreamParser initialized")
    
    def extract_tool_calls(self, data: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from a delta message."""
        logger.debug(f"Extracting tool calls from message: {json.dumps(data, indent=2)}")
        tool_calls = []
        
        if (choices := data.get('choices')) and choices:
            first_choice = choices[0]
            if (delta := first_choice.get('delta')) and 'tool_calls' in delta:
                logger.info(f"Found tool_calls in delta: {json.dumps(delta['tool_calls'], indent=2)}")
                for tool_call in delta['tool_calls']:
                    try:
                        logger.debug(f"Processing tool call: {json.dumps(tool_call, indent=2)}")
                        arguments = json.loads(tool_call['function']['arguments'])
                        tool_call_obj = ToolCall(
                            id=tool_call['id'],
                            name=tool_call['function']['name'],
                            arguments=arguments
                        )
                        logger.info(f"Created ToolCall object: {tool_call_obj}")
                        tool_calls.append(tool_call_obj)
                    except Exception as e:
                        logger.error(f"Error parsing tool call: {str(e)}", exc_info=True)
                        logger.error(f"Problematic tool call data: {json.dumps(tool_call, indent=2)}")
        
        return tool_calls

    async def process_chunk(self, chunk: bytes) -> Optional[Dict[str, Any]]:
        """Parse and process a chunk, executing any tool calls found."""
        logger.debug(f"Processing chunk of size: {len(chunk)} bytes")
        parsed = self.parse_chunk(chunk)
        if not parsed:
            logger.debug("No valid JSON found in chunk")
            return None
        
        tool_calls = self.extract_tool_calls(parsed)
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls")
            results = []
            for tool_call in tool_calls:
                result = await self.tool_executor.execute_tool(tool_call)
                logger.info(f"Tool execution result: {json.dumps(result, indent=2)}")
                results.append(result)
            
            # Create a tool result message in OpenAI format
            response = {
                "id": parsed['id'],
                "object": "chat.completion.chunk",
                "created": parsed['created'],
                "model": parsed['model'],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": results
                    },
                    "finish_reason": None
                }]
            }
            logger.debug(f"Created tool response message: {json.dumps(response, indent=2)}")
            return response
        
        return parsed

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
                
                # Check for tool calls in the OpenAI streaming format
                if (choices := data.get('choices')) and choices:
                    first_choice = choices[0]
                    if (delta := first_choice.get('delta')) and 'tool_calls' in delta:
                        logger.info(f"Found tool calls in delta: {json.dumps(delta['tool_calls'], indent=2)}")
                
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
                    
                    # Check for tool calls in the OpenAI streaming format
                    if (choices := data.get('choices')) and choices:
                        first_choice = choices[0]
                        if (delta := first_choice.get('delta')) and 'tool_calls' in delta:
                            logger.info(f"Found tool calls in buffered delta: {json.dumps(delta['tool_calls'], indent=2)}")
                    
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
                    processed = await parser.process_chunk(chunk)
                    if processed:
                        # Re-encode as SSE format
                        yield f"data: {json.dumps(processed)}\n\n".encode('utf-8')
                    else:
                        # Forward unparseable chunks as-is
                        yield chunk
    
    logger.info("Returning streaming response")
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
    
