"""Phase 2 tests: path safety, MCP server tools, the MCP adapter, skills, agent loop.

All offline: notes live in a temp folder, the MCP client talks to the server
IN MEMORY (no child process), and the agent gets a scripted fake LLM.
"""

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from documind import config
from documind.agent.skills import (
    SkillFormatError,
    find_skills,
    make_load_skill_tool,
    parse_skill,
    skills_catalog,
)
from documind.agent.tool_agent import STOPPED_MESSAGE, run_agent, run_tool
from documind.mcp_server import server
from documind.mcp_server.mcp_client import mcp_tools, to_langchain_tool
from documind.mcp_server.safe_paths import UnsafePathError, resolve_inside

# ---------------------------------------------------------------- helpers


@pytest.fixture
def notes(tmp_path, monkeypatch) -> Path:
    """A temp notes folder; the server tools read NOTES_DIR from settings."""
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "sql.md").write_text("# SQL\nSELECT picks columns.", encoding="utf-8")
    (tmp_path / "samples" / "ignore.txt").write_text("not a note", encoding="utf-8")
    (tmp_path / "secret.env").write_text("KEY=123", encoding="utf-8")
    monkeypatch.setenv("NOTES_DIR", str(tmp_path))
    config.get_settings.cache_clear()  # re-read settings with the new NOTES_DIR
    return tmp_path


def write_skill(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True)
    path = folder / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


GOOD_SKILL = "---\nname: cards\ndescription: Make cards. Use for revision.\n---\n# Cards\nStep 1: read."


class ScriptedLLM(BaseChatModel):
    """Fake chat model: returns the scripted replies in order and records
    every message list it was sent, so tests can check what the LLM saw."""

    replies: list[AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self  # a real model would attach the tool schemas here

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        # After the script runs out, keep repeating the last reply.
        reply = self.replies[min(len(self.seen), len(self.replies)) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def ask_tool(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AI reply that asks for one tool call (what a real LLM sends back)."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ---------------------------------------------------------------- safe paths


def test_resolve_inside_allows_normal_names(tmp_path):
    assert resolve_inside(tmp_path, "samples/sql.md") == (tmp_path / "samples" / "sql.md").resolve()
    assert resolve_inside(tmp_path, "samples/../a.md") == (tmp_path / "a.md").resolve()


@pytest.mark.parametrize(
    "name",
    ["", "   ", "../.env", "samples/../../.env", r"C:\Windows\win.ini", "/etc/passwd", "."],
)
def test_resolve_inside_blocks_escapes(tmp_path, name):
    with pytest.raises(UnsafePathError):
        resolve_inside(tmp_path, name)


# ---------------------------------------------------------------- MCP server tools
# @mcp.tool returns the plain function, so tools can be tested like any function.


def test_list_notes_only_md_and_pdf(notes):
    assert server.list_notes() == ["samples/sql.md"]


def test_read_note_returns_text(notes):
    assert "SELECT" in server.read_note("samples/sql.md")


@pytest.mark.parametrize(
    "name, message",
    [
        ("../outside.md", "escapes"),
        ("secret.env", "only .md and .pdf"),
        ("samples/missing.md", "list_notes"),
    ],
)
def test_read_note_errors(notes, name, message):
    with pytest.raises(ToolError, match=message):
        server.read_note(name)


def test_save_note_writes_only_to_output(notes):
    assert server.save_note("cards.md", "Q/A") == "output/cards.md"
    assert (notes / "output" / "cards.md").read_text(encoding="utf-8") == "Q/A"


def test_save_note_refuses_overwrite_unless_asked(notes):
    server.save_note("cards.md", "v1")
    with pytest.raises(ToolError, match="already exists"):
        server.save_note("cards.md", "v2")
    server.save_note("cards.md", "v2", overwrite=True)
    assert (notes / "output" / "cards.md").read_text(encoding="utf-8") == "v2"


@pytest.mark.parametrize("name", ["cards.txt", "../samples/sql.md", "../../evil.md"])
def test_save_note_rejects_bad_names(notes, name):
    with pytest.raises(ToolError):
        server.save_note(name, "x", overwrite=True)
    # The user's own note is untouched.
    assert (notes / "samples" / "sql.md").read_text(encoding="utf-8").startswith("# SQL")


def test_wiki_needs_contact_info(monkeypatch):
    monkeypatch.setenv("WIKI_CONTACT", "")
    config.get_settings.cache_clear()
    with pytest.raises(ToolError, match="WIKI_CONTACT"):
        server.wiki_user_agent()


# ---------------------------------------------------------------- MCP protocol + adapter
# Client(server.mcp) = in-memory transport: real MCP messages, no child process.


def test_mcp_client_sees_four_tools(notes):
    async def names():
        async with Client(server.mcp) as client:
            return sorted(t.name for t in await client.list_tools())

    assert asyncio.run(names()) == ["list_notes", "read_note", "save_note", "wiki_search"]


def test_adapter_calls_tool_and_turns_errors_into_text(notes):
    async def call(name, args):
        async with Client(server.mcp) as client:
            mcp_tool = next(t for t in await client.list_tools() if t.name == name)
            return await to_langchain_tool(client, mcp_tool).ainvoke(args)

    assert "SELECT" in asyncio.run(call("read_note", {"name": "samples/sql.md"}))
    assert '"samples/sql.md"' in asyncio.run(call("list_notes", {}))  # list -> JSON text
    assert asyncio.run(call("read_note", {"name": "../x.md"})).startswith("ERROR:")


# ---------------------------------------------------------------- skills


def test_parse_skill(tmp_path):
    skill = parse_skill(write_skill(tmp_path / "cards", GOOD_SKILL))
    assert skill.name == "cards"
    assert skill.description == "Make cards. Use for revision."
    assert skill.instructions == "# Cards\nStep 1: read."


@pytest.mark.parametrize(
    "text",
    [
        "# no frontmatter",
        "---\nname: x\ndescription: y\n",  # never closed
        "---\nname: x\n---\nbody",  # no description
    ],
)
def test_parse_skill_rejects_bad_frontmatter(tmp_path, text):
    with pytest.raises(SkillFormatError):
        parse_skill(write_skill(tmp_path / "bad", text))


def test_project_skills_are_valid():
    # Real files in skills/: catches a broken SKILL.md before the agent does.
    assert {"flashcards", "summary"} <= set(find_skills())


def test_catalog_has_descriptions_not_instructions(tmp_path):
    write_skill(tmp_path / "cards", GOOD_SKILL)
    catalog = skills_catalog(find_skills(tmp_path))
    assert catalog == "- cards: Make cards. Use for revision."  # progressive disclosure
    assert skills_catalog({}) == "(no skills available)"


def test_load_skill_tool(tmp_path):
    write_skill(tmp_path / "cards", GOOD_SKILL)
    load_skill = make_load_skill_tool(find_skills(tmp_path))
    assert load_skill.invoke({"name": "cards"}) == "# Cards\nStep 1: read."
    assert load_skill.invoke({"name": "nope"}).startswith("ERROR: no skill called 'nope'")


# ---------------------------------------------------------------- agent loop


@tool
def shout(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


@pytest.fixture
def skills(tmp_path):
    write_skill(tmp_path / "cards", GOOD_SKILL)
    return find_skills(tmp_path)


def test_agent_runs_tools_then_answers(skills):
    llm = ScriptedLLM(
        replies=[
            ask_tool("load_skill", {"name": "cards"}, "c1"),
            ask_tool("shout", {"text": "hi"}, "c2"),
            AIMessage(content="All done."),
        ],
    )
    result = asyncio.run(run_agent("go", [shout], skills, llm=llm))

    assert result.answer == "All done." and result.finished
    assert result.steps == ['load_skill({"name": "cards"})', 'shout({"text": "hi"})']
    # The system prompt carried the catalog, and the last call saw both tool results.
    assert "- cards: Make cards." in llm.seen[0][0].content
    tool_results = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_results] == [
        ("c1", "# Cards\nStep 1: read."),
        ("c2", "HI"),
    ]


def test_agent_reports_unknown_tool_and_continues(skills):
    llm = ScriptedLLM(replies=[ask_tool("delete_everything", {}), AIMessage(content="ok")])
    result = asyncio.run(run_agent("go", [shout], skills, llm=llm))
    assert result.answer == "ok"
    assert "ERROR: there is no tool called 'delete_everything'" in llm.seen[-1][-1].content


def test_agent_stops_at_step_limit(skills):
    llm = ScriptedLLM(replies=[ask_tool("sho ut", {"text": "again"})])  # never stops
    result = asyncio.run(run_agent("go", [shout], skills, llm=llm, max_steps=3))
    assert not result.finished
    assert result.answer == STOPPED_MESSAGE
    assert len(result.steps) == 3


def test_run_tool_turns_exceptions_into_text():
    @tool
    def broken(x: int) -> int:
        """Always fails."""
        raise RuntimeError("disk on fire")

    assert asyncio.run(run_tool({"broken": broken}, "broken", {"x": 1})) == (
        "ERROR: RuntimeError: disk on fire"
    )


# ---------------------------------------------------------------- live


@pytest.mark.live
def test_live_agent_over_stdio():
    """Real Groq + the real MCP server started as a child process."""

    async def ask():
        async with mcp_tools() as tools:
            return await run_agent("Which study notes do I have?", tools, find_skills())

    result = asyncio.run(ask())
    assert result.finished
    assert any(step.startswith("list_notes") for step in result.steps)
    assert "sql_basics" in result.answer.lower().replace(" ", "_")