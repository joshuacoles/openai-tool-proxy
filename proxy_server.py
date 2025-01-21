from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import json
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator

from openai.types import FunctionDefinition
from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import Function

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define test tool using FunctionDefinition
CALCULATOR_TOOL = FunctionDefinition(
    name="calculate",
    description="Perform basic arithmetic calculations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform",
            },
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["operation", "a", "b"],
    },
)

# Define our tools to inject
PROXY_TOOLS = [{"type": "function", "function": CALCULATOR_TOOL.model_dump()}]

app = FastAPI(
    debug=True,  # Enable debug mode for more detailed logging
    title="Tool Proxy",
    description="A proxy server that adds tool definitions to LLM requests",
)


class CalculatorTool:
    @staticmethod
    async def execute(args: Dict[str, Any]) -> Any:
        operation = args["operation"]
        a = args["a"]
        b = args["b"]

        logger.info(f"Executing calculator tool: {operation}({a}, {b})")

        result = {
            "add": lambda: a + b,
            "subtract": lambda: a - b,
            "multiply": lambda: a * b,
            "divide": lambda: a / b if b != 0 else "Error: Division by zero",
        }[operation]()

        logger.info(f"Calculator result: {result}")
        return result


class ToolExecutor:
    def __init__(self):
        self.tools = {"calculate": CalculatorTool.execute}
        logger.info(f"ToolExecutor initialized with tools: {list(self.tools.keys())}")

    async def execute_tool(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> ChatCompletionToolMessageParam:
        """Execute a tool call and return the result in the expected format."""
        logger.info(
            f"Executing tool call: id={tool_call.id}, name={tool_call.function.name}, args={tool_call.function.arguments}"
        )

        if tool_call.function.name not in self.tools:
            logger.error(
                f"Unknown tool: {tool_call.function.name}, available tools: {list(self.tools.keys())}"
            )
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=f"Error: Unknown tool {tool_call.function.name}",
            )

        try:
            logger.debug(
                f"Invoking tool {tool_call.function.name} with arguments: {tool_call.function.arguments}"
            )
            arguments = json.loads(tool_call.function.arguments)
            result = await self.tools[tool_call.function.name](arguments)
            logger.info(
                f"Tool execution successful: id={tool_call.id}, result={result}"
            )

            return ChatCompletionToolMessageParam(
                role="tool", tool_call_id=tool_call.id, content=str(result)
            )
        except Exception as e:
            logger.error(
                f"Tool execution error for {tool_call.function.name}: {str(e)}",
                exc_info=True,
            )
            return ChatCompletionToolMessageParam(
                role="tool", tool_call_id=tool_call.id, content=f"Error: {str(e)}"
            )


class StreamParser:
    def __init__(self, client: httpx.AsyncClient):
        self.buffer = ""
        self.message_count = 0
        self.tool_executor = ToolExecutor()
        self.client = client
        logger.info("StreamParser initialized")

    def extract_tool_calls(
        self, data: Dict[str, Any]
    ) -> List[ChatCompletionMessageToolCall]:
        """Extract tool calls from a delta message."""
        logger.debug(
            f"Extracting tool calls from message: {json.dumps(data, indent=2)}"
        )
        tool_calls = []

        if (choices := data.get("choices")) and choices:
            first_choice = choices[0]
            if (delta := first_choice.get("delta")) and "tool_calls" in delta:
                logger.info(
                    f"Found tool_calls in delta: {json.dumps(delta['tool_calls'], indent=2)}"
                )
                for tool_call in delta["tool_calls"]:
                    try:
                        logger.debug(
                            f"Processing tool call: {json.dumps(tool_call, indent=2)}"
                        )
                        tool_call_obj = ChatCompletionMessageToolCall(
                            id=tool_call["id"],
                            type="function",
                            function=Function(
                                name=tool_call["function"]["name"],
                                arguments=tool_call["function"]["arguments"],
                            ),
                        )
                        logger.info(f"Created ToolCall object: {tool_call_obj}")
                        tool_calls.append(tool_call_obj)
                    except Exception as e:
                        logger.error(
                            f"Error parsing tool call: {str(e)}", exc_info=True
                        )
                        logger.error(
                            f"Problematic tool call data: {json.dumps(tool_call, indent=2)}"
                        )

        return tool_calls

    async def send_tool_results_to_ollama(
        self,
        original_message: Dict[str, Any],
        tool_results: List[ChatCompletionToolMessageParam],
    ) -> AsyncGenerator[bytes, None]:
        """Send tool results back to Ollama as a new message and stream the response."""
        logger.info("Sending tool results back to Ollama")

        # Create the request to send back to Ollama
        request_body = {
            "model": original_message["model"],
            "messages": tool_results,  # Send tool results as messages
            "stream": True,
            "tools": PROXY_TOOLS,  # Include tools definition for potential follow-up calls
        }

        logger.debug(
            f"Sending tool results to Ollama: {json.dumps(request_body, indent=2)}"
        )

        try:
            async with self.client.stream(
                "POST", "http://localhost:11434/v1/chat/completions", json=request_body
            ) as response:
                logger.info("Receiving streamed response from tool result submission")
                async for chunk in response.aiter_bytes():
                    logger.debug(f"Received follow-up chunk: {chunk.decode()}")
                    yield chunk

        except Exception as e:
            logger.error(
                f"Error sending tool results to Ollama: {str(e)}", exc_info=True
            )
            raise

    async def process_chunk(self, chunk: bytes) -> AsyncGenerator[bytes, None]:
        """Parse and process a chunk, executing any tool calls found."""
        logger.debug(f"Processing chunk of size: {len(chunk)} bytes")
        parsed = self.parse_chunk(chunk)
        if not parsed:
            logger.debug("No valid JSON found in chunk")
            yield chunk
            return

        tool_calls = self.extract_tool_calls(parsed)
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls")
            results: List[ChatCompletionToolMessageParam] = []
            for tool_call in tool_calls:
                result = await self.tool_executor.execute_tool(tool_call)
                logger.info(
                    f"Tool execution result: {result}"
                )
                results.append(result)

            # Send results back to Ollama and stream the response
            try:
                # First yield the tool execution results to the client
                response = {
                    "id": parsed["id"],
                    "object": "chat.completion.chunk",
                    "created": parsed["created"],
                    "model": parsed["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [result for result in results]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(response)}\n\n".encode("utf-8")

                # Then stream Ollama's response to the tool results
                async for follow_up_chunk in self.send_tool_results_to_ollama(
                    parsed, results
                ):
                    yield follow_up_chunk

            except Exception as e:
                logger.error("Failed to send tool results to Ollama", exc_info=True)
                error_response = {
                    "id": parsed["id"],
                    "object": "chat.completion.chunk",
                    "created": parsed["created"],
                    "model": parsed["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"Error processing tool results: {str(e)}"
                            },
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_response)}\n\n".encode("utf-8")
        else:
            # No tool calls, just forward the chunk
            yield f"data: {json.dumps(parsed)}\n\n".encode("utf-8")

    def parse_chunk(self, chunk: bytes) -> Optional[Dict[str, Any]]:
        """Parse a chunk of bytes into a JSON object."""
        try:
            text = chunk.decode("utf-8")
            logger.debug(f"Raw chunk text: {text}")

            # Handle server-sent events format
            if text.startswith("data: "):
                text = text.removeprefix("data: ")

            # Skip empty chunks
            if not text.strip():
                logger.debug("Skipping empty chunk")
                return None

            # Try to parse as JSON
            try:
                data = json.loads(text)
                self.message_count += 1
                logger.debug(
                    f"Successfully parsed message {self.message_count}: {json.dumps(data, indent=2)}"
                )

                # Check for tool calls in the OpenAI streaming format
                if (choices := data.get("choices")) and choices:
                    first_choice = choices[0]
                    if (delta := first_choice.get("delta")) and "tool_calls" in delta:
                        logger.info(
                            f"Found tool calls in delta: {json.dumps(delta['tool_calls'], indent=2)}"
                        )

                return data

            except json.JSONDecodeError:
                # Add to buffer and try to parse
                self.buffer += text
                logger.debug(f"Added to buffer. Current buffer: {self.buffer}")

                try:
                    data = json.loads(self.buffer)
                    self.buffer = ""  # Clear buffer on successful parse
                    self.message_count += 1
                    logger.debug(
                        f"Successfully parsed buffered message {self.message_count}: {json.dumps(data, indent=2)}"
                    )

                    # Check for tool calls in the OpenAI streaming format
                    if (choices := data.get("choices")) and choices:
                        first_choice = choices[0]
                        if (
                            delta := first_choice.get("delta")
                        ) and "tool_calls" in delta:
                            logger.info(
                                f"Found tool calls in buffered delta: {json.dumps(delta['tool_calls'], indent=2)}"
                            )

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
    stream = body.get("stream", False)
    logger.info(f"Stream mode: {stream}")

    timeout = httpx.Timeout(10.0, connect=10.0)

    if not stream:
        logger.info("Handling non-streaming request")
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Sending request to Ollama")
            ollama_response = await client.post(
                "http://localhost:11434/v1/chat/completions", json=body
            )
            response_body = ollama_response.json()
            logger.info(
                f"Received response from Ollama: {json.dumps(response_body, indent=2)}"
            )
            return Response(
                content=ollama_response.content,
                status_code=ollama_response.status_code,
                headers=dict(ollama_response.headers),
            )

    # Handle streaming response
    logger.info("Handling streaming request")

    async def stream_response():
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Starting streaming request to Ollama")
            parser = StreamParser(client)

            async with client.stream(
                "POST", "http://localhost:11434/v1/chat/completions", json=body
            ) as response:
                logger.info("Established streaming connection with Ollama")
                async for chunk in response.aiter_bytes():
                    async for processed_chunk in parser.process_chunk(chunk):
                        yield processed_chunk

    logger.info("Returning streaming response")
    return StreamingResponse(stream_response(), media_type="text/event-stream")
