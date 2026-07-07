def get_prompt(cfg):
    return f"""Write a fluent, self-contained English text about the following topic.

Constraints:
- Use your own wording and structure.
- Do not refer to the existence of a source text.
- Do not include headings or bullet points.
- Keep the length between {cfg.min_words} and {cfg.max_words} words.
- Return only the generated text.

Topic:
{cfg.topic}
"""