"""Hermes evaluation benchmark via `hermes -z` CLI (bypasses broken Fabric native API on Windows).

The NeMo Fabric native API spawns the Hermes adapter as a subprocess via `_native.pyd`,
which fails on Windows with os error 267 (invalid directory name) — an NVIDIA bug.
This script uses `hermes -z` directly, which works.

Usage:
    python hermes_eval_cli.py --help
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERMES_EXE = r"C:\Users\nicec\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe"


def run_case(prompt: str, case_id: str) -> dict:
    """Run a single test case via `hermes -z` and return structured result."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [HERMES_EXE, "-z", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
        )
        elapsed = time.perf_counter() - start

        output = result.stdout.strip()
        error = result.stderr.strip() if result.returncode != 0 else None

        return {
            "case_id": case_id,
            "status": "ok" if result.returncode == 0 else "error",
            "elapsed_seconds": round(elapsed, 3),
            "output": output,
            "error": error,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "case_id": case_id,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 3),
            "output": "",
            "error": "Timeout after 120s",
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "case_id": case_id,
            "status": "error",
            "elapsed_seconds": round(elapsed, 3),
            "output": "",
            "error": str(exc),
        }


def load_cases(cases_file: str | None) -> list[dict]:
    """Load test cases from JSONL/JSON, or return defaults."""
    if cases_file:
        path = Path(cases_file)
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))

    return [
        {
            "id": "smoke-001",
            "prompt": "Скажи привет одним словом.",
            "check_contains": "Привет",
        },
        {
            "id": "math-002",
            "prompt": "Сколько будет 25 умножить на 17?",
            "check_contains": "425",
        },
        {
            "id": "translate-003",
            "prompt": "Переведи на английский: 'Утро вечера мудренее'.",
            "check_contains": "morning",
        },
        {
            "id": "code-004",
            "prompt": "Напиши функцию на Python, которая проверяет, простое ли число.",
            "check_contains": "def",
        },
        {
            "id": "reasoning-005",
            "prompt": "Если у меня 3 яблока, я съел одно, сколько осталось?",
            "check_contains": "2",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes QA benchmark via `hermes -z` CLI")
    parser.add_argument("--cases", default=None, help="Path to test cases JSON/JSONL file")
    parser.add_argument("--out", default="hermes_eval_results.json", help="Output JSON file")
    parser.add_argument("--out-csv", default=None, help="Optional CSV output file")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = []

    print(f"Running {len(cases)} cases via hermes -z ...\n")

    for case in cases:
        case_id = case.get("id", "unknown")
        prompt = case.get("prompt", "")
        print(f"  [{case_id}] {prompt[:60]}{'...' if len(prompt) > 60 else ''}", end=" ... ")

        result = run_case(prompt, case_id)

        check = case.get("check_contains")
        passed = None
        if check:
            passed = check.lower() in result["output"].lower()
            status_marker = "PASS" if passed else "FAIL"
        else:
            status_marker = result["status"].upper()

        print(f"{status_marker} ({result['elapsed_seconds']}s)")

        results.append({
            **result,
            "prompt": prompt,
            "check_contains": check,
            "check_passed": passed,
        })

    # Save JSON
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path.resolve()}")

    # Save CSV if requested
    if args.out_csv:
        csv_path = Path(args.out_csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "case_id", "status", "elapsed_seconds", "prompt", "output",
                "error", "check_contains", "check_passed",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"CSV written to {csv_path.resolve()}")

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    passed = sum(1 for r in results if r["check_passed"] is True)
    print(f"\nSummary: {ok}/{total} OK, {err}/{total} errors, {passed}/{total} checks passed")


if __name__ == "__main__":
    main()
