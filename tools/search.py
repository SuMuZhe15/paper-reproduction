
import re
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "knowledge.csv"

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "ReActResearchProject/1.0 (educational research)"
}

session = requests.Session()
session.headers.update(HEADERS)


def search_knowledge(query: str) -> str:
    """搜索本地知识库，保留原有功能。"""
    if not DATA_PATH.exists():
        return f"知识库文件不存在：{DATA_PATH}"

    df = pd.read_csv(DATA_PATH)

    if not {"keyword", "content"}.issubset(df.columns):
        return "知识库格式错误，需要 keyword 和 content 两列。"

    query = query.strip().lower()

    if not query:
        return "查询内容不能为空。"

    matched = df[
        df["keyword"].fillna("").astype(str).str.lower().apply(
            lambda keyword: keyword in query or query in keyword
        )
    ]

    if matched.empty:
        return "未找到相关知识。"

    return "\n".join(
        f"关键词：{row['keyword']}\n内容：{row['content']}"
        for _, row in matched.head(3).iterrows()
    )


def search_wikipedia(query: str, limit: int = 3) -> str:
    """搜索英文 Wikipedia，返回候选页面标题和摘要。"""
    query = query.strip()

    if not query:
        return "查询内容不能为空。"

    try:
        response = session.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as exc:
        return f"Wikipedia 搜索失败：{type(exc).__name__}: {exc}"

    results = data.get("query", {}).get("search", [])

    if not results:
        return "Wikipedia 未找到相关页面。"

    output = []

    for item in results:
        title = item.get("title", "")
        snippet = re.sub(
            r"<[^>]+>", "", item.get("snippet", "")
        )
        snippet = (
            snippet.replace("&quot;", '"')
            .replace("&#039;", "'")
            .replace("&amp;", "&")
        )

        output.append(
            f"标题：{title}\n"
            f"摘要：{snippet}\n"
            f"页面：https://en.wikipedia.org/wiki/"
            f"{requests.utils.quote(title.replace(' ', '_'))}"
        )

    return "\n\n".join(output)


def open_wikipedia_page(title: str) -> str:
    """读取指定 Wikipedia 页面正文的纯文本提取结果。"""
    title = title.strip()

    if not title:
        return "页面标题不能为空。"

    try:
        response = session.get(
            WIKI_API,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exchars": 5000,
                "redirects": 1,
                "titles": title,
                "format": "json",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as exc:
        return f"Wikipedia 页面读取失败：{type(exc).__name__}: {exc}"

    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        if page.get("missing") is not None:
            return f"未找到 Wikipedia 页面：{title}"

        extract = page.get("extract", "").strip()

        if extract:
            return (
                f"页面标题：{page.get('title', title)}\n"
                f"页面链接：https://en.wikipedia.org/wiki/"
                f"{requests.utils.quote(page.get('title', title).replace(' ', '_'))}\n\n"
                f"{extract}"
            )

    return f"Wikipedia 页面没有可读取的正文：{title}"


if __name__ == "__main__":
    query = "ReAct reasoning and acting language models"

    print("=== Wikipedia 搜索测试 ===")
    print(search_wikipedia(query))

    print("\n=== Wikipedia 页面读取测试 ===")
    print(open_wikipedia_page("ReAct (language model)"))
