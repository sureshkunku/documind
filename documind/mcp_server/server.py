"""The DocuMind MCP server: four note tools any MCP client can use.

CONCEPTS
- MCP SERVER: a small program that offers TOOLS over a standard protocol.
  Clients (our agent, Claude Desktop, IDEs) ask it "what tools do you
  have?" and then "call tool X with these arguments".
- FastMCP builds the tool description from your Python function:
    function name   -> tool name
    docstring       -> description the LLM reads to decide WHEN to use it
    type hints      -> JSON schema of the arguments
  So docstrings here are not just comments: they ARE the instructions.
- TRANSPORT "stdio": the client starts this file as a child process and
  talks to it through stdin/stdout. No ports, no web server.
  (So never print() in a stdio server: stdout is the protocol channel.)
- ToolError: a clean error message sent back to the LLM, so it can
  correct itself (e.g. pick another file name) instead of crashing.
"""

from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from documind.config import get_settings
from documind.mcp_server.safe_paths import UnsafePathError, resolve_inside
from documind.rag.loader import SUPPORTED_SUFFIXES, read_text

MAX_READ_CHARS = 20_000  # keep a huge PDF from flooding the LLM's context
WIKI_API = "https://en.wikipedia.org/w/api.php"

mcp = FastMCP(
    name="documind-notes",
    instructions="Tools to list, read and save the user's study notes, and to look up facts on Wikipedia.",
)


def notes_dir() -> Path:
    return get_settings().notes_dir


def output_dir() -> Path:
    return notes_dir() / "output"


@mcp.tool
def list_notes() -> list[str]:
    """List all note files (.md and .pdf) as paths relative to the notes folder.
    Files the assistant saved earlier are under 'output/'."""
    root = notes_dir()
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(p.relative_to(root).as_posix() for p in files)


@mcp.tool
def read_note(name: str) -> str:
    """Return the full text of one note. `name` is a path from list_notes(),
    e.g. 'samples/sql_basics.md'."""
    try:
        path = resolve_inside(notes_dir(), name)
    except UnsafePathError as error:
        raise ToolError(str(error)) from error
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ToolError(f"only .md and .pdf notes can be read, not {name!r}")
    if not path.is_file():
        raise ToolError(f"no note called {name!r}. Call list_notes() to see the available names.")

    text = read_text(path)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n\n[... cut at {MAX_READ_CHARS} characters]"
    return text


@mcp.tool
def save_note(name: str, text: str, overwrite: bool = False) -> str:
    """Save markdown text as a new note in the 'output/' folder, e.g. name='cards.md'.
    Refuses to replace an existing file unless overwrite=True. Returns the saved path."""
    if not name.lower().endswith(".md"):
        raise ToolError(f"notes are saved as markdown: the name must end with .md, got {name!r}")
    try:
        path = resolve_inside(output_dir(), name)  # output/ only: agents never edit your own notes
    except UnsafePathError as error:
        raise ToolError(str(error)) from error
    if path.exists() and not overwrite:
        raise ToolError(f"{name!r} already exists. Choose another name, or pass overwrite=True.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.relative_to(notes_dir().resolve()).as_posix()


def wiki_user_agent() -> str:
    """Wikipedia's robot policy: API clients must send a User-Agent with
    contact info (an email or website), or requests are refused (HTTP 403)."""
    contact = get_settings().wiki_contact
    if not contact:
        raise ToolError(
            "wiki_search is not set up: Wikipedia requires contact info. "
            "Add WIKI_CONTACT=<your email or website> to .env."
        )
    return f"DocuMind/0.1 ({contact}) httpx/{httpx.__version__}"


@mcp.tool
def wiki_search(query: str) -> list[dict]:
    """Search English Wikipedia. Returns up to 3 results with title, a short
    summary and the page URL. Use it to check facts that are not in the notes."""
    params = {
        # One request does both steps: search for pages, then fetch their intros.
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 3,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exsentences": 3,
        "inprop": "url",
    }
    try:
        response = httpx.get(WIKI_API, params=params, headers={"User-Agent": wiki_user_agent()}, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ToolError(f"Wikipedia request failed: {error}") from error

    pages = response.json().get("query", {}).get("pages", {}).values()
    # "index" is the search rank; the API returns pages in no particular order.
    ranked = sorted(pages, key=lambda page: page.get("index", 0))
    return [
        {"title": page["title"], "summary": page.get("extract", ""), "url": page.get("fullurl", "")}
        for page in ranked
    ]


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio transport by default