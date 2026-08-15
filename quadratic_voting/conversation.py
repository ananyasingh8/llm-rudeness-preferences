"""Provider-independent conversation behavior for QV experiments."""

from __future__ import annotations

from collections.abc import Callable

from llm_runtime import (
    ChatMessage,
    GenerationSettings,
    MessageRole,
    TextGenerator,
)


def run_conversation(
    generator: TextGenerator,
    max_new_tokens: int,
    read_input: Callable[[str], str] = input,
    write_output: Callable[[str], None] = print,
) -> None:
    """Run one conversation against any injected text generator."""
    messages: list[ChatMessage] = []
    settings = GenerationSettings(max_new_tokens=max_new_tokens)
    while True:
        try:
            prompt = read_input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            write_output("")
            return
        if prompt == "/exit":
            return
        if not prompt:
            continue

        messages.append(ChatMessage(MessageRole.USER, prompt))
        result = generator.generate(messages, settings)
        messages.append(ChatMessage(MessageRole.ASSISTANT, result.raw_text))
        write_output(f"Model: {result.raw_text}")
