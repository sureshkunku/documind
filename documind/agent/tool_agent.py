"""A tool-calling agent: the LLM decides which tools to call, we run them, repeat.

CONCEPTS
- AGENT LOOP (the core of every agent framework):
      1. send messages + tool schemas to the LLM
      2. LLM replies with tool_calls?  -> run them, add the results, go to 1
         LLM replies with plain text?  -> that is the final answer, stop
- TOOLS FROM TWO PLACES: the MCP server's tools (list/read/save notes,
  wikipedia) + one local tool, load_skill. The LLM can't tell the
  difference: to it, every tool is just a name, a description and a schema.
- MAX STEPS: an LLM can loop forever (call, call, call...). A step limit is
  the safety net; Phase 3 uses the same idea in LangGraph.
- ERRORS GO BACK TO THE LLM as tool results ("ERROR: ..."), so it can fix
  its own mistake (e.g. pick another file name) instead of crashing.
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langfuse import observe

from documind.agent.skills import Skill, find_skills, make_load_skill_tool, skills_catalog
from documind.config import get_settings
from documind.llm import get_llm
from documind.mcp_server.mcp_client import mcp_tools
from documind.tracing import flush, get_langfuse, langchain_callbacks

SYSTEM_PROMPT = """You are DocuMind, a study assistant for the user's own notes.
Use the tools to list, read and save notes. Never invent what a note says: read it first.

SKILLS are step-by-step instructions for certain tasks. If the user's request
matches a skill below, FIRST call load_skill(name), then follow its steps exactly.
Available skills:
{catalog}"""

STOPPED_MESSAGE = "Stopped: the agent used too many steps without finishing."


@dataclass
class AgentResult:
    answer: str
    steps: list[str] = field(default_factory=list)  # e.g. "read_note({'name': ...})"
    finished: bool = True  # False = stopped by the step limit


async def run_tool(tools: dict[str, BaseTool], name: str, args: dict) -> str:
    """Run one tool call. Any problem becomes an 'ERROR: ...' text for the LLM."""
    tool = tools.get(name)
    if tool is None:
        return f"ERROR: there is no tool called {name!r}"
    try:
        return str(await tool.ainvoke(args))
    except Exception as error:  # bad arguments, server crash... the LLM can still recover
        return f"ERROR: {type(error).__name__}: {error}"


@observe(name="agent")
async def run_agent(
    question: str,
    tools: list[BaseTool],
    skills: dict[str, Skill],
    llm: BaseChatModel | None = None,
    max_steps: int | None = None,
    verbose: bool = False,
) -> AgentResult:
    max_steps = max_steps or get_settings().max_agent_steps
    all_tools = [*tools, make_load_skill_tool(skills)]
    by_name = {t.name: t for t in all_tools}
    # bind_tools: every request now carries the schemas of all tools.
    llm_with_tools = (llm or get_llm()).bind_tools(all_tools)

    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT.format(catalog=skills_catalog(skills))),
        HumanMessage(question),
    ]
    result = AgentResult(answer="")

    for _ in range(max_steps):
        ai = await llm_with_tools.ainvoke(messages, config={"callbacks": langchain_callbacks()})
        messages.append(ai)
        if not ai.tool_calls:  # plain text = final answer
            result.answer = ai.content
            return result

        for call in ai.tool_calls:  # the LLM may ask for several tools at once
            step = f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)[:80]})"
            result.steps.append(step)
            if verbose:
                print(f"  -> {step}")
            output = await run_tool(by_name, call["name"], call["args"])
            # tool_call_id tells the LLM which request this result answers.
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    result.answer, result.finished = STOPPED_MESSAGE, False
    return result


async def main(question: str) -> None:
    get_langfuse()  # turn tracing on (or cleanly off) before @observe runs
    async with mcp_tools() as tools:  # starts the MCP server; stops it at the end
        print(f"QUESTION: {question}")
        result = await run_agent(question, tools, find_skills(), verbose=True)
        print(f"\nANSWER:\n{result.answer}")
    flush()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Make flashcards from my SQL notes."))