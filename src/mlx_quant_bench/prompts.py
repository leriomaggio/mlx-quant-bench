"""Benchmark prompt suite.

The prompts span three difficulty buckets to make precision-induced quality
degradation visible. Models tend to fail differently at different precisions:
factual recall holds up well even at 3-bit, while multi-step reasoning and
careful instruction-following start breaking earlier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A single benchmark prompt.

    Attributes:
        id: Stable identifier used in the JSONL output and reports.
        category: One of ``"factual"``, ``"reasoning"``, ``"instruction"``.
            Used to group results in the report.
        text: The prompt content. Will be wrapped in the model's chat template
            at benchmark time.
    """
    id: str
    category: str
    text: str


PROMPTS: list[Prompt] = [
    # --- Factual recall: easy, robust to quantization -----------------------
    Prompt(
        id="capital_france",
        category="factual",
        text="What is the capital of France? Answer in one sentence.",
    ),
    Prompt(
        id="speed_of_light",
        category="factual",
        text="What is the approximate speed of light in vacuum? Give the value in km/s.",
    ),
    Prompt(
        id="shakespeare_birth",
        category="factual",
        text="In what year was William Shakespeare born?",
    ),

    # --- Multi-step reasoning: shows precision degradation first ------------
    Prompt(
        id="train_problem",
        category="reasoning",
        text=(
            "Two trains leave stations 300 km apart, travelling toward each "
            "other. One goes 60 km/h, the other 90 km/h. How long until they "
            "meet? Show your reasoning step by step."
        ),
    ),
    Prompt(
        id="age_puzzle",
        category="reasoning",
        text=(
            "Alice is twice as old as Bob. In 5 years, the sum of their ages "
            "will be 40. How old is Bob now? Show your work."
        ),
    ),
    Prompt(
        id="probability",
        category="reasoning",
        text=(
            "I roll two fair six-sided dice. What is the probability that the "
            "sum is exactly 7? Explain your reasoning."
        ),
    ),

    # --- Instruction-following: format-sensitive, breaks visibly at 3-bit ---
    Prompt(
        id="json_extract",
        category="instruction",
        text=(
            "Extract the name and age from this sentence as JSON with keys "
            '"name" and "age": "Yesterday I met Alice, who is 32 years old, '
            'at a coffee shop." Return ONLY valid JSON, no other text.'
        ),
    ),
    Prompt(
        id="bullet_summary",
        category="instruction",
        text=(
            "Summarise the following in exactly 3 bullet points, no more, no "
            "less:\n\nThe Industrial Revolution transformed manufacturing in "
            "the 18th and 19th centuries. It began in Britain and spread "
            "worldwide. Steam power, mechanised textile production, and the "
            "rise of the factory system were defining features."
        ),
    ),
    Prompt(
        id="haiku_format",
        category="instruction",
        text=(
            "Write a haiku about quantization. Strict 5-7-5 syllable structure. "
            "Output the haiku only, no commentary."
        ),
    ),
    Prompt(
        id="negation",
        category="instruction",
        text=(
            "List 3 things that are NOT mammals. Output as a numbered list. "
            "Do not list any mammals."
        ),
    ),
]


def by_category(category: str) -> list[Prompt]:
    """Return all prompts in a given category."""
    return [p for p in PROMPTS if p.category == category]


def categories() -> list[str]:
    """All distinct categories present in the prompt set."""
    seen: list[str] = []
    for p in PROMPTS:
        if p.category not in seen:
            seen.append(p.category)
    return seen
