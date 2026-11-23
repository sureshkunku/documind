---
name: flashcards
description: Turn a study note into question-and-answer flashcards for revision. Use when the user asks for flashcards, quiz cards or Q&A practice.
---
# Flashcards

Follow these steps exactly.

1. Find the note: call list_notes(), pick the note that matches the topic, then read_note() it.
2. Write 6 to 10 flashcards using ONLY facts from that note. Never add outside facts.
3. One card per idea. Format every card like this:

   **Q:** a short, specific question
   **A:** the answer in one or two sentences

4. Start the file with a title line: `# Flashcards: <topic>` and a line `Source: <note path>`.
5. Save with save_note(name="<topic>_flashcards.md", text=...).
   If the name already exists, add a number, e.g. `<topic>_flashcards_2.md`.
6. Reply with how many cards you made and the saved path. Do not repeat all cards.