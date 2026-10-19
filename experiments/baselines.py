
import re

from agent.llm import ask_llm
from tools.search import search_wikipedia, open_wikipedia_page
from agent.react import run_react as run_react_agent

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


def run_standard(question: str) -> str:
    return ask_llm(f"请直接回答下面的问题。\n问题：{question}")


def run_cot(question: str) -> str:
    prompt = (
        "请先分析问题，再给出最终答案。不要调用外部工具。\n"
        f"问题：{question}"
    )
    return ask_llm(prompt)


def run_act(question: str) -> str:
    """Act：选择并执行一次工具调用，再根据观察结果回答。"""
    _reset_stats()

    prompt = (
        "你可以调用 Wikipedia 工具获取信息。"
        "请只选择一个工具，并严格按格式输出：\n"
        "Action: search_wikipedia[查询内容]\n"
        "或 Action: open_wikipedia_page[页面标题]\n"
        f"问题：{question}"
    )

    response = ask_llm(prompt)
    print(f"\n--- Act Action ---\n{response}")

    match = re.search(
        r"Action:\s*(search_wikipedia|open_wikipedia_page)\s*\[(.*?)\]",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if not match or not match.group(2).strip():
        raise RuntimeError(f"Act 未生成有效 Action：{response}")

    tool_name = match.group(1).lower()
    tool_input = match.group(2).strip()

    LAST_RUN_STATS["steps"] = 1
    LAST_RUN_STATS["tool_calls"] = 1

    try:
        if tool_name == "search_wikipedia":
            LAST_RUN_STATS["search_calls"] = 1
            observation = search_wikipedia(tool_input)
        else:
            LAST_RUN_STATS["open_page_calls"] = 1
            observation = open_wikipedia_page(tool_input)
    except Exception as exc:
        LAST_RUN_STATS["tool_errors"] = 1
        observation = f"工具异常：{type(exc).__name__}: {exc}"

    print(f"\n--- Act Observation ---\n{observation}")

    answer_prompt = (
        "请依据工具返回的信息回答问题。证据不足时明确说明，不要编造。\n"
        f"问题：{question}\n"
        f"Observation：{observation}"
    )
    return ask_llm(answer_prompt)


def run_react(question: str) -> str:
    return run_react_agent(question)


METHODS = {
    "Standard": run_standard,
    "CoT": run_cot,
    "Act": run_act,
    "ReAct": run_react,
}


def get_stats(method_name: str) -> dict:
    if method_name == "Act":
        return LAST_RUN_STATS.copy()

    if method_name == "ReAct":
        from agent import react
        return react.LAST_RUN_STATS.copy()

    return {
        "steps": 0,
        "tool_calls": 0,
        "search_calls": 0,
        "open_page_calls": 0,
        "tool_errors": 0,
    }


if __name__ == "__main__":
    question = "ReAct 中 Thought、Action 和 Observation 分别是什么？"

    for name, method in METHODS.items():
        print(f"\n{'=' * 12} {name} {'=' * 12}")
        try:
            print(method(question))
            print("统计：", get_stats(name))
        except Exception as exc:
            print(f"运行失败：{type(exc).__name__}: {exc}")
