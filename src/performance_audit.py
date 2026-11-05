"""Compare Python-list and NumPy salary adjustments and audit array layouts."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def median_runtime(operation, repeats: int) -> tuple[float, object]:
    durations = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), result


def main() -> None:
    employee_count = 1_000_000
    repeats = 5
    salaries = [50_000.0 + (index % 50_000) for index in range(employee_count)]
    salary_array = np.asarray(salaries, dtype=np.float64)

    list_seconds, list_result = median_runtime(
        lambda: [salary * 1.10 for salary in salaries], repeats
    )
    numpy_seconds, numpy_result = median_runtime(
        lambda: salary_array * 1.10, repeats
    )
    if not np.allclose(list_result, numpy_result):
        raise RuntimeError("List and NumPy salary calculations produced different results.")

    inventory = np.arange(1, 25, dtype=np.int64).reshape(4, 6)
    weekly_temperatures = np.array(
        [
            [18, 19, 17, 20, 21, 18, 16],
            [22, 23, 21, 24, 25, 22, 20],
            [15, 16, 14, 17, 18, 15, 13],
        ],
        dtype=np.float64,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": {
            "operation": "Apply a 10% salary increase to 1,000,000 employees",
            "repeats": repeats,
            "list_median_seconds": list_seconds,
            "numpy_median_seconds": numpy_seconds,
            "time_efficiency_gap_list_over_numpy": list_seconds / numpy_seconds,
            "results_match": True,
        },
        "array_audit": {
            "inventory": {
                "shape": list(inventory.shape),
                "ndim": int(inventory.ndim),
                "size": int(inventory.size),
                "dtype": str(inventory.dtype),
                "nbytes": int(inventory.nbytes),
                "warehouse_totals_axis_1": inventory.sum(axis=1).tolist(),
                "product_totals_axis_0": inventory.sum(axis=0).tolist(),
            },
            "weekly_temperatures": {
                "shape": list(weekly_temperatures.shape),
                "ndim": int(weekly_temperatures.ndim),
                "size": int(weekly_temperatures.size),
                "dtype": str(weekly_temperatures.dtype),
                "daily_averages_axis_0": weekly_temperatures.mean(axis=0).tolist(),
                "city_averages_axis_1": weekly_temperatures.mean(axis=1).tolist(),
            },
        },
    }
    report_path = REPORT_DIR / "performance_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved audit report to {report_path}")


if __name__ == "__main__":
    main()
