"""Keep every AI-chosen file path inside one allowed folder.

CONCEPTS
- The LLM picks the file names it sends to our tools. A confused model,
  or a note containing hidden instructions ("prompt injection"), could
  ask for "../../.env" and leak your API keys.
- RULE: never trust a path from the AI. Resolve it to a full real path,
  then check it is still inside the allowed folder.
- resolve() also follows ".." and symlinks, so tricks like
  "samples/../../x" are caught after resolving, not by string matching.
"""

from pathlib import Path


class UnsafePathError(ValueError):
    """The requested file name points outside the allowed folder."""


def resolve_inside(base: Path, name: str) -> Path:
    """Return base/name as a full path, or raise UnsafePathError if it escapes base."""
    if not name or not name.strip():
        raise UnsafePathError("file name is empty")

    candidate = Path(name)
    # anchor is "C:\", "/", "\\server\share\" or "C:" — any of them means
    # "start somewhere else than base", so reject before even resolving.
    if candidate.anchor:
        raise UnsafePathError(f"absolute paths are not allowed: {name!r}")

    base = base.resolve()
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise UnsafePathError(f"path escapes the allowed folder: {name!r}")
    if resolved == base:
        raise UnsafePathError(f"name must point to a file, not the folder itself: {name!r}")
    return resolved


if __name__ == "__main__":
    from documind.config import get_settings

    base = get_settings().notes_dir
    attempts = [
        "samples/sql_basics.md",  # normal
        "samples/../samples/sql_basics.md",  # odd but still inside -> allowed
        "../.env",  # escape to your secrets
        "samples/../../.env",  # sneaky escape
        r"C:\Windows\win.ini",  # absolute Windows path
        "/etc/passwd",  # absolute Unix-style path
        ".",  # the folder itself
    ]
    for name in attempts:
        try:
            path = resolve_inside(base, name)
            print(f"ALLOWED  {name!r:40} -> {path.relative_to(base.resolve()).as_posix()}")
        except UnsafePathError as error:
            print(f"BLOCKED  {name!r:40} {error}")