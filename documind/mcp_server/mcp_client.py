"""Connect to the MCP server over stdio and turn its tools into LangChain tools.

CONCEPTS
- STDIO TRANSPORT: the client STARTS the server as a child process
  (python -m documind.mcp_server.server) and exchanges MCP messages
  through its stdin/stdout. When the client closes, the server stops.
- ADAPTER: LangChain agents only understand LangChain tools. An adapter
  wraps each MCP tool as a LangChain StructuredTool:
      MCP tool name         -> tool name
      MCP description       -> description (what the LLM reads)
      MCP input_schema      -> args_schema (JSON schema, used as-is)
      "call it"             -> an async function that asks the server
  langchain-mcp-adapters does exactly this; we write it ourselves because
  that library doesn't support the mcp 2.x that fastmcp 4 uses yet.
- ERRORS AS TEXT: a server ToolError becomes "ERROR: ..." text returned
  to the LLM, so it can read the message and try something else.
"""

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from langchain_core.tools import StructuredTool

from documind.config import PROJECT_ROOT


def server_transport() -> StdioTransport:
    return StdioTransport(
        # sys.executable = this project's .venv python, so the server sees the same packages.
        command=sys.executable,
        args=["-m", "documind.mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )


def to_text(result) -> str:
    """Tool results go back to the LLM as text; JSON keeps lists/dicts readable."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


def to_langchain_tool(client: Client, mcp_tool) -> StructuredTool:
    async def call_on_server(**arguments) -> str:
        try:
            result = await client.call_tool(mcp_tool.name, arguments)
        except ToolError as error:
            return f"ERROR: {error}"
        return to_text(result.data)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.input_schema,  # JSON schema straight from the server
        coroutine=call_on_server,  # async: MCP calls are network-style I/O
    )


@asynccontextmanager
async def mcp_tools() -> AsyncIterator[list[StructuredTool]]:
    """Start the server, yield its tools as LangChain tools, stop the server on exit.

    Usage:
        async with mcp_tools() as tools:
            ...use tools while the server is running...
    """
    async with Client(server_transport()) as client:
        mcp_tool_list = await client.list_tools()
        yield [to_langchain_tool(client, t) for t in mcp_tool_list]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        async with mcp_tools() as tools:
            print("LangChain tools loaded from the MCP server (over stdio):")
            for t in tools:
                print(f"  {t.name}  args={list(t.args)}")

            by_name = {t.name: t for t in tools}
            # ainvoke = async invoke. Same as the LLM would trigger, but called by hand.
            print("\nlist_notes ->", await by_name["list_notes"].ainvoke({}))
            print("\nread_note('../.env') ->", await by_name["read_note"].ainvoke({"name": "../.env"}))
            text = await by_name["read_note"].ainvoke({"name": "samples/python_basics.md"})
            print("\nread_note('samples/python_basics.md') ->", text[:70].replace("\n", " "), "...")

    asyncio.run(main())