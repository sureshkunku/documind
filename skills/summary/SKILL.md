---
name: summary
description: Write a short revision summary of a study note. Use when the user asks to summarise, recap or give key points of a note or topic.
---
# Revision summary

Follow these steps exactly.

1. Find the note: call list_notes(), pick the matching note, then read_note() it.
2. Write the summary using ONLY facts from that note:
   - `# Summary: <topic>` and `Source: <note path>`
   - **Key points**: 3 to 6 bullets, one line each
   - **Remember**: the single most important idea, one sentence
3. Keep it under 150 words.
4. Only save it if the user asked to save it: save_note(name="<topic>_summary.md", text=...).
5. Reply with the summary (and the saved path if you saved it).