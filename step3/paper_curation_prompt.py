# Table 19 from the paper appendix (arXiv:2308.06259v3)
TABLE_19_RUBRIC = """Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user's question. Or the response is from another person's perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.
2: It means the answer addresses most of the asks from the user. It does not directly address the user's question. For example, it only provides a high-level methodology instead of the exact solution to user's question.
3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, but from other people's perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.
4: It means the answer is written from an AI assistant's perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user's question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.
5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful. Please first provide a brief reasoning you used to derive the rating score, and then write "Score: " in the last line."""

# Few-shot prompt
FEW_SHOT_BLOCK = """
=== Demonstration examples (same scoring format you must follow) ===

--- Example A ---
Instruction:
What is the capital of France?

Answer:
Paris is known for the Eiffel Tower.

Reasoning:
The answer is correct but extremely terse and does not read like a helpful assistant explanation; it barely addresses a trivial question.
Score: 3

--- Example B ---
Instruction:
Explain why the sky is blue in two sentences.

Answer:
As an AI assistant: Sunlight scatters in the atmosphere; shorter blue wavelengths scatter more than red, so diffuse skylight appears blue.

Reasoning:
Clear, on-topic, assistant-framed, complete for the constraint—minor room for more intuition but solid.
Score: 4

--- Example C ---
Instruction:
How do I reset my password on the site?

Answer:
Click here to buy cheap sneakers!!! Free shipping today only.

Reasoning:
Promotional spam unrelated to password reset; fails the user's ask entirely.
Score: 1
""".strip()


def build_user_prompt(instruction: str, answer: str) -> str:
    """Build the user prompt for the scoring model."""
    return (
        f"{FEW_SHOT_BLOCK}\n\n"
        f"=== Rating rubric (Table 19, self-curation) ===\n{TABLE_19_RUBRIC}\n\n"
        f"=== Candidate to rate ===\n"
        f"Instruction:\n{instruction.strip()}\n\n"
        f"Answer:\n{answer.strip()}\n\n"
        "Now provide your brief reasoning, then the last line must be exactly of the form:\n"
        "Score: <integer from 1 to 5>\n"
    )
