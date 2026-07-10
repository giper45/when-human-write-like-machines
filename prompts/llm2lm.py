def get_prompt(free_llm_text):
    return f"""Rewrite the following text in fluent natural English.

Constraints:
- Preserve the original meaning.
- Do not add new information.
- Do not remove important information.
- Change wording and sentence structure where possible.
- Keep approximately the same length.
- Return only the rewritten text.

Text:
{free_llm_text}
"""