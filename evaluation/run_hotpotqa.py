
import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from experiments import baselines

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hotpotqa_250.json"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "hotpotqa_results.csv"

FIELDS = [
    "run_id", "timestamp", "question_id", "question",
    "answer_gold", "type", "level", "method", "status",
    "answer_pred", "error", "elapsed_seconds",
    "steps", "tool_calls", "search_calls",
    "open_page_calls", "tool_errors",
]


def load_completed():
    """读取已经成功完成的题目—方法组合，用于断点续跑。"""
    completed = set()

    if not OUTPUT_PATH.exists():
        return completed

    with OUTPUT_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question_id") and row.get("method"):
                completed.add((
        str(row["question_id"]),
        row["method"],
    ))
    return completed


def run_experiment():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "找不到 data/hotpotqa_500.json。"
            "请先运行 python -m data.download_hotpotqa"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        with OUTPUT_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            old_fields = next(csv.reader(f), [])

        if old_fields != FIELDS:
            raise RuntimeError(
                "hotpotqa_results.csv 表头与当前版本不一致。"
                "请备份并移走该结果文件后再运行。"
            )

    completed = load_completed()
    run_id = str(uuid.uuid4())[:8]

    new_file = (
        not OUTPUT_PATH.exists()
        or OUTPUT_PATH.stat().st_size == 0
    )

    total = len(questions) * len(baselines.METHODS)
    finished = len(completed)

    print(f"数据集：{len(questions)} 题")
    print(f"方法：{', '.join(baselines.METHODS.keys())}")
    print(f"计划组合数：{total}")
    print(f"已完成组合：{finished}")
    print(f"本批次 ID：{run_id}")

    with OUTPUT_PATH.open(
        "a", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        for index, item in enumerate(questions, start=1):
            qid = str(item["id"])

            for name, method in baselines.METHODS.items():
                if (qid, name) in completed:
                    continue

                print(
                    f"\n[{index}/{len(questions)}] "
                    f"{name} | {qid}"
                )
                print("问题：", item["question"])

                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "question_id": qid,
                    "question": item["question"],
                    "answer_gold": item["answer"],
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "method": name,
                    "status": "success",
                    "answer_pred": "",
                    "error": "",
                    "elapsed_seconds": 0,
                    "steps": 0,
                    "tool_calls": 0,
                    "search_calls": 0,
                    "open_page_calls": 0,
                    "tool_errors": 0,
                }

                start = time.perf_counter()

                try:
                    answer = method(item["question"])
                    row["answer_pred"] = answer or ""
                    print("回答：", row["answer_pred"])

                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print("运行失败：", row["error"])

                finally:
                    row["elapsed_seconds"] = round(
                        time.perf_counter() - start, 3
                    )

                try:
                    stats = baselines.get_stats(name)
                except Exception:
                    stats = {}

                for key in (
                    "steps", "tool_calls", "search_calls",
                    "open_page_calls", "tool_errors",
                ):
                    row[key] = stats.get(key, 0)

                writer.writerow(row)
                f.flush()

                if row["status"] == "success":
                    completed.add((qid, name))

                finished += 1
                print(
                    f"进度：{finished}/{total}，"
                    f"耗时：{row['elapsed_seconds']} 秒"
                )

    print("\n实验运行结束。")
    print(f"原始结果：{OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()
