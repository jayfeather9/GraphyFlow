import subprocess
import os
import sys
import re
import csv
from datetime import datetime
from pathlib import Path

# 数据集列表配置 - 在这里添加你需要测试的数据集
DATASETS = [
    "/data/feiyang/test/test/datasets/rmat-19-32.txt",
    "/data/feiyang/test/test/datasets/rmat-21-32.txt",
    "/data/feiyang/test/test/datasets/rmat-24-16.txt",
]

# 测试目标（hw, hw_emu, sw_emu）
TARGET = "hw"

# 输出文件配置
LOG_DIR = "./benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_OUTPUT = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def setup_environment():
    os.system("bash -c 'source env.sh && make exe'")


def get_cmd(dataset: str):
    log_name = dataset.split("/")[-1].replace(".txt", "").replace(" ", "_")
    cmd = f"bash run.sh {TARGET} {dataset} > {LOG_DIR}/{log_name}.log 2>&1"
    return cmd


def extract_metrics(log_content: str):
    metrics = {}
    for line in log_content.split("\n"):
        if "Total FPGA Kernel Execution Time" in line:
            metrics["total_time"] = float(line.split(" ")[-2])
        if "Total MTEPS (Edges / Total Time)" in line:
            metrics["total_mteps"] = float(line.split(" ")[-2])
    return metrics


def benchmark():
    results = []
    for dataset in DATASETS:
        cmd = get_cmd(dataset)
        os.system(cmd)
        print(cmd)

        log_name = dataset.split("/")[-1].replace(".txt", "").replace(" ", "_")
        log_file = f"{LOG_DIR}/{log_name}.log"
        with open(log_file, "r") as f:
            log_content = f.read()

        # extract metrics from log
        metrics = extract_metrics(log_content)
        metrics["dataset"] = dataset
        results.append(metrics)

    with open(CSV_OUTPUT, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "total_time", "total_mteps"])
        for result in results:
            writer.writerow([result["dataset"], result["total_time"], result["total_mteps"]])


if __name__ == "__main__":
    setup_environment()
    benchmark()
