"""Talk to the MCP server the way any client does: list tools, call them.

CONCEPTS
- Client(mcp) connects IN-MEMORY: same MCP messages, no subprocess.
  Great for learning and tests. Step 2.4 connects over stdio instead.
- MCP is async (clients may call many tools at once), so we use
  async/await and asyncio.run().
- A ToolError raised in the server arrives here as an exception with
  the server's message — that's what the LLM will see in Step 2.5.
"""

import asyncio

from fastmcp import Client

from documind.mcp_server.server import mcp


async def main() -> None:
    async with Client(mcp) as client:
        print("TOOLS THE SERVER OFFERS:")
        for tool in await client.list_tools():
            args = list(tool.inputSchema.get("properties", {}))
            print(f"  {tool.name}({', '.join(args)})  — {tool.description.splitlines()[0]}")

        print("\nlist_notes ->", (await client.call_tool("list_notes", {})).data)

        text = (await client.call_tool("read_note", {"name": "samples/sql_basics.md"})).data
        print("read_note  ->", text[:60].replace("\n", " "), "...")

        for bad_name in ["../.env", "samples/missing.md"]:
            try:
                await client.call_tool("read_note", {"name": bad_name})
            except Exception as error:
                print(f"read_note({bad_name!r}) -> ERROR: {error}")

        print("\nwiki_search('Attention Is All You Need'):")
        try:
            for item in (await client.call_tool("wiki_search", {"query": "Attention Is All You Need"})).data:
                print(f"  {item['title']}  {item['url']}")
        except Exception as error:
            print("  ERROR:", error)


if __name__ == "__main__":
    asyncio.run(main())