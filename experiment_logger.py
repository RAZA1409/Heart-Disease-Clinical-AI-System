import csv
import os
from datetime import datetime

RESULTS_FILE = "results.csv"

def log_experiment(method, accuracy, params):
    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Method",
                "Accuracy",
                "C",
                "Gamma",
                "Kernel"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            method,
            round(accuracy * 100, 2),
            params.get("C", "default"),
            params.get("gamma", "default"),
            params.get("kernel", "default")
        ])