"""Agent Skills: reusable how-to instructions the agent loads only when needed.

CONCEPTS
- SKILL: a folder with a SKILL.md file. The top (between --- lines) is
  "frontmatter" with a name and a description; the rest is the full
  step-by-step instructions for one kind of task.
- PROGRESSIVE DISCLOSURE: the system prompt lists only each skill's name +
  description (a few lines). The full instructions are loaded with the
  load_skill tool when the LLM decides a skill fits the task. So 50 skills
  cost almost nothing until one is actually used.
- Skills vs tools: a TOOL does something (read a file, call an API).
  A SKILL teaches HOW to do a task, usually by using several tools.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

from documind.config import get_settings


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str  # the SKILL.md text after the frontmatter
    path: Path


class SkillFormatError(ValueError):
    """Raised when a SKILL.md has no valid frontmatter."""


def parse_skill(path: Path) -> Skill:
    """Read one SKILL.md: '---' / 'key: value' lines / '---' / instructions."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError(f"{path}: must start with a '---' frontmatter line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise SkillFormatError(f"{path}: frontmatter is not closed with '---'") from None

    fields = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    if not fields.get("name") or not fields.get("description"):
        raise SkillFormatError(f"{path}: frontmatter needs 'name' and 'description'")

    instructions = "\n".join(lines[end + 1 :]).strip()
    return Skill(fields["name"], fields["description"], instructions, path)


def find_skills(skills_dir: Path | None = None) -> dict[str, Skill]:
    """Every skills/<folder>/SKILL.md, keyed by skill name."""
    skills_dir = skills_dir or get_settings().skills_dir
    skills = [parse_skill(p) for p in sorted(skills_dir.glob("*/SKILL.md"))]
    return {s.name: s for s in skills}


def skills_catalog(skills: dict[str, Skill]) -> str:
    """The short list that goes into the system prompt (names + descriptions only)."""
    if not skills:
        return "(no skills available)"
    return "\n".join(f"- {s.name}: {s.description}" for s in skills.values())


def make_load_skill_tool(skills: dict[str, Skill]) -> StructuredTool:
    """A normal LangChain tool (not MCP): skills live with the agent, not the server."""

    @tool
    def load_skill(name: str) -> str:
        """Load the full step-by-step instructions of a skill by its name.
        Call this BEFORE starting a task that matches a skill in the list."""
        skill = skills.get(name)
        if skill is None:
            return f"ERROR: no skill called {name!r}. Available: {', '.join(skills) or 'none'}"
        return skill.instructions

    return load_skill


if __name__ == "__main__":
    found = find_skills()
    print("Catalog (what the system prompt shows):\n" + skills_catalog(found))
    for skill in found.values():
        print(f"\nload_skill({skill.name!r}) would return {len(skill.instructions)} characters:")
        print("  " + skill.instructions.splitlines()[0])