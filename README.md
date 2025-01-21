# OpenAI Tool Proxy Server

This is a proxy server that allows you to transparently insert tool definitions into OpenAI requests and fulfills tool calls from the response.

## Goals

- Insert tool definitions transparently into the request
- Parse tool results from the response
- Inject tool results into the response
- Ignore tool calls for tools which are not handled by the proxy server.
- Handle streaming and non-streaming responses
- Offer tools from connected MCP servers
