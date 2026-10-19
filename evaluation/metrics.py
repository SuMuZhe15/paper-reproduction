

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "baseline_results.csv"
QUESTIONS = ROOT / "data" / "questions.json"


def evaluate_results():
    if not INPUT.exists():
        print("未找到实验结果，请先运行 run_experiment.py。")
        return

    df = pd.read_csv(INPUT)

    if df.empty:
        print("实验结果为空。")
        return

    required = [
        "question_id", "method", "status",
        "answer", "elapsed_seconds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"结果文件缺少字段：{missing}")
        return

    if not QUESTIONS.exists():
        print("未找到 questions.json。")
        return

    with QUESTIONS.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    keyword_map = {
        int(q["id"]): q.get("expected_keywords", [])
        for q in questions
    }

    df["question_id"] = pd.to_numeric(
        df["question_id"], errors="coerce"
    )
    df["elapsed_seconds"] = pd.to_numeric(
        df["elapsed_seconds"], errors="coerce"
    )

    metric_columns = [
        "steps", "tool_calls", "search_calls",
        "open_page_calls", "tool_errors",
    ]

    for col in metric_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    # 关键词严格通过率 + 关键词覆盖率
    def score(row):
        if row["status"] != "success":
            return pd.Series({
                "answer_correct": 0,
                "keyword_coverage": 0.0,
            })

        if pd.isna(row["question_id"]):
            return pd.Series({
                "answer_correct": float("nan"),
                "keyword_coverage": float("nan"),
            })

        keywords = keyword_map.get(
            int(row["question_id"]), []
        )

        if not keywords:
            return pd.Series({
                "answer_correct": float("nan"),
                "keyword_coverage": float("nan"),
            })

        answer = str(row["answer"]).lower()

        matched = sum(
            str(keyword).lower() in answer
            for keyword in keywords
        )

        coverage = matched / len(keywords)

        return pd.Series({
            "answer_correct": int(matched == len(keywords)),
            "keyword_coverage": coverage,
        })

    df[["answer_correct", "keyword_coverage"]] = df.apply(
        score, axis=1
    )

    def summarize(groups):
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
            average_seconds=("elapsed_seconds", "mean"),
            keyword_pass_rate=("answer_correct", "mean"),
            average_keyword_coverage=("keyword_coverage", "mean"),
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

        result["keyword_pass_rate_pct"] = (
            result["keyword_pass_rate"] * 100
        )

        result["average_keyword_coverage_pct"] = (
            result["average_keyword_coverage"] * 100
        )

        # 没有工具调用时错误率记为 0%，避免 NaN
        result["tool_error_rate_pct"] = (
            result["total_tool_errors"]
            / result["total_tool_calls"].replace(0, float("nan"))
            * 100
        )
        result.loc[
            result["total_tool_calls"] == 0,
            "tool_error_rate_pct",
        ] = 0.0

        result = result.drop(columns=[
            "keyword_pass_rate",
            "average_keyword_coverage",
        ])

        numeric = result.select_dtypes(
            include="number"
        ).columns
        result[numeric] = result[numeric].round(2)

        return result

    overall = summarize(["method"])
    overall.to_csv(
        RESULTS / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if "run_id" in df.columns:
        per_run = summarize(["run_id", "method"])
        per_run.to_csv(
            RESULTS / "metrics_by_run.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n=== 四种方法总体对比 ===")
    print(overall.to_string(index=False))

    print("\n已输出：results/metrics_summary.csv")

    if "run_id" in df.columns:
        print("已输出：results/metrics_by_run.csv")

    print(
        "\n指标说明："
        "\n- keyword_pass_rate_pct：全部关键词都命中的样本比例。"
        "\n- average_keyword_coverage_pct：平均关键词覆盖比例。"
        "\n- tool_error_rate_pct：工具调用中的错误比例；"
        "没有工具调用时显示 0%。"
        "\n注意：关键词指标是启发式评估，不等同于语义正确率。"
    )


if __name__ == "__main__":
    evaluate_results()
