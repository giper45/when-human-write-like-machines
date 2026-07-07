def get_prompt(cfg):
    return f"""Rewrite the following text in fluent natural English.
Constraints:
- Preserve the original meaning.
- Preserve factual claims, entities, and relationships.
- Do not add new information.
- Do not remove important information.
- Change wording and sentence structure where possible.
- Keep approximately the same length.
- Return only the rewritten text.

Text:
{cfg.source_text}
"""