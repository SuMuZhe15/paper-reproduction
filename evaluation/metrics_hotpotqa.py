
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "hotpotqa_results.csv"


def normalize_answer(text):
    """HotpotQA 风格的英文答案规范化。"""
    text = str(text).lower()
    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


def token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def add_scores(df):
    df["exact_match"] = df.apply(
        lambda row: (
            exact_match(
                row["answer_pred"], row["answer_gold"]
            )
            if row["status"] == "success"
            else 0
        ),
        axis=1,
    )

    df["token_f1"] = df.apply(
        lambda row: (
            token_f1(
                row["answer_pred"], row["answer_gold"]
            )
            if row["status"] == "success"
            else 0.0
        ),
        axis=1,
    )

    for col in (
        "elapsed_seconds", "steps", "tool_calls",
        "search_calls", "open_page_calls", "tool_errors",
    ):
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    return df


def summarize(df, groups):
    result = df.groupby(
        groups, dropna=False
    ).agg(
        total_runs=("status", "size"),
        successful_runs=(
            "status",
            lambda s: (s == "success").sum(),
        ),
        failed_runs=(
            "status",
            lambda s: (s != "success").sum(),
        ),
        EM=("exact_match", "mean"),
        F1=("token_f1", "mean"),
        average_seconds=("elapsed_seconds", "mean"),
        average_steps=("steps", "mean"),
        average_tool_calls=("tool_calls", "mean"),
        average_search_calls=("search_calls", "mean"),
        average_open_page_calls=("open_page_calls", "mean"),
        total_tool_calls=("tool_calls", "sum"),
        total_tool_errors=("tool_errors", "sum"),
    ).reset_index()

    result["success_rate_pct"] = (
        result["successful_runs"]
        / result["total_runs"] * 100
    )
    result["EM_pct"] = result["EM"] * 100
    result["F1_pct"] = result["F1"] * 100

    result["tool_error_rate_pct"] = (
        result["total_tool_errors"]
        / result["total_tool_calls"].replace(0, float("nan"))
        * 100
    )
    result.loc[
        result["total_tool_calls"] == 0,
        "tool_error_rate_pct",
    ] = 0.0

    result = result.drop(columns=["EM", "F1"])

    numeric = result.select_dtypes(
        include="number"
    ).columns
    result[numeric] = result[numeric].round(2)

    return result


def main():
    if not INPUT.exists():
        print("未找到结果，请先运行 python -m evaluation.run_hotpotqa")
        return

    df = pd.read_csv(INPUT)

    if df.empty:
        print("结果文件为空。")
        return

    required = {
        "question_id", "method", "status",
        "answer_pred", "answer_gold",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"结果文件缺少字段：{sorted(missing)}")

    # 如果失败样本重跑后产生多条记录，保留该题该方法的最后一次结果
    df = df.drop_duplicates(
        subset=["question_id", "method"],
        keep="last",
    ).copy()

    df = add_scores(df)

    overall = summarize(df, ["method"])
    overall.to_csv(
        RESULTS / "hotpotqa_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== HotpotQA 总体结果 ===")
    print(overall.to_string(index=False))

    if "type" in df.columns:
        by_type = summarize(df, ["method", "type"])
        by_type.to_csv(
            RESULTS / "hotpotqa_metrics_by_type.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n=== 按问题类型分组 ===")
        print(by_type.to_string(index=False))

    if "level" in df.columns:
        by_level = summarize(df, ["method", "level"])
        by_level.to_csv(
            RESULTS / "hotpotqa_metrics_by_level.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n评估文件已输出至 results/。")
    print("EM：规范化后完全匹配率。")
    print("F1：预测答案与标准答案的 token 重合 F1。")


if __name__ == "__main__":
    main()
