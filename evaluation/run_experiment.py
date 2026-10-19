

import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from experiments import baselines

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.json"
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "baseline_results.csv"

FIELDS = [
    "run_id", "timestamp", "question_id", "question", "method",
    "status", "answer", "error", "elapsed_seconds",
    "steps", "tool_calls", "search_calls", "open_page_calls",
    "tool_errors",
]


def run_experiment():
    RESULTS_DIR.mkdir(exist_ok=True)

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    run_id = str(uuid.uuid4())[:8]

    # 新版字段与旧 CSV 表头不一致时，停止追加，避免破坏数据
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            old_fields = next(csv.reader(f), [])

        if old_fields != FIELDS:
            raise RuntimeError(
                "现有 baseline_results.csv 表头与新版不一致。"
                "请先备份旧文件，再将其改名或移出 results 目录后重试。"
            )

    new_file = not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        for item in questions:
            for name, method in baselines.METHODS.items():
                print(f"\n{'=' * 10} {name} | Q{item['id']} {'=' * 10}")

                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "question_id": item["id"],
                    "question": item["question"],
                    "method": name,
                    "status": "success",
                    "answer": "",
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
                    row["answer"] = method(item["question"])
                    print("答案：", row["answer"])
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print("错误：", row["error"])
                finally:
                    row["elapsed_seconds"] = round(
                        time.perf_counter() - start, 3
                    )

                stats = baselines.get_stats(name)
                for key in (
                    "steps", "tool_calls", "search_calls",
                    "open_page_calls", "tool_errors",
                ):
                    row[key] = stats.get(key, 0)

                writer.writerow(row)
                f.flush()

    print(f"\n批次 ID：{run_id}")
    print(f"原始结果：{OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiment()
