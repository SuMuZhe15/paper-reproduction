
import re

from agent.llm import ask_llm
from tools.search import search_wikipedia, open_wikipedia_page


SYSTEM_PROMPT = """
You are a ReAct agent.

Available tools:
1. search_wikipedia(query): search English Wikipedia.
2. open_wikipedia_page(title): read a Wikipedia page.

For each turn, output exactly one of these formats.

Tool call:
Thought: Briefly explain what information is needed.
Action: search_wikipedia[query]

Or:
Thought: Briefly explain what information is needed.
Action: open_wikipedia_page[page title]

Final answer:
Thought: Briefly state the basis for the conclusion.
Final Answer: Your answer

Rules:
- Use tools when external evidence is needed.
- Use Observation to decide the next step.
- Never invent tool results.
- The final response must start with Final Answer:.
"""


LAST_RUN_STATS = {}


def _reset_stats():
    global LAST_RUN_STATS
    LAST_RUN_STATS = {
        "steps": 0,
        "tool_calls": 0,
        "search_calls": 0,
        "open_page_calls": 0,
        "tool_errors": 0,
    }


def run_react(question: str, max_steps: int = 5) -> str:
    """Run the ReAct loop and record execution metrics."""
    _reset_stats()
    history = f"Question: {question}\n"

    for step in range(max_steps):
        LAST_RUN_STATS["steps"] += 1

        response = ask_llm(
            history + "\nContinue with the next step.",
            system_prompt=SYSTEM_PROMPT,
        )

        print(f"\n--- ReAct Step {step + 1} ---")
        print(response)

        final_match = re.search(
            r"Final Answer:\s*(.*)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match and final_match.group(1).strip():
            return final_match.group(1).strip()

        action_match = re.search(
            r"Action:\s*(search_wikipedia|open_wikipedia_page)"
            r"\s*\[(.*?)\]",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if not action_match:
            history += (
                f"\nAssistant: {response}\n"
                "Observation: Invalid format. Output a valid Action "
                "or Final Answer and try again.\n"
            )
            continue

        tool_name = action_match.group(1).lower()
        tool_input = action_match.group(2).strip()

        if not tool_input:
            history += (
                f"\nAssistant: {response}\n"
                "Observation: Empty tool input. Try again.\n"
            )
            continue

        LAST_RUN_STATS["tool_calls"] += 1

        try:
            if tool_name == "search_wikipedia":
                LAST_RUN_STATS["search_calls"] += 1
                observation = search_wikipedia(tool_input)
            else:
                LAST_RUN_STATS["open_page_calls"] += 1
                observation = open_wikipedia_page(tool_input)
        except Exception as exc:
            LAST_RUN_STATS["tool_errors"] += 1
            observation = f"Tool error: {type(exc).__name__}: {exc}"

        print(f"\nObservation: {observation}")
        history += (
            f"\nAssistant: {response}\n"
            f"Observation: {observation}\n"
        )

    raise RuntimeError(
        f"ReAct reached max_steps={max_steps} without a final answer."
    )


if __name__ == "__main__":
    question = (
        "What is ReAct in language models, and how does it combine "
        "reasoning and acting?"
    )
    try:
        print("\n=== Final Answer ===")
        print(run_react(question))
    except RuntimeError as exc:
        print(exc)
    print("\n=== Execution Metrics ===")
    print(LAST_RUN_STATS)

