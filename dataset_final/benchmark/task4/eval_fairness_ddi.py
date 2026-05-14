#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
from collections import defaultdict


CSV_PATH = "dataset_final/ddi/ddi_metadata_isic_format_step1.1.csv"


TASK4_DIR = "dataset_final/benchmark/task4"

def normalize_option(s):
    s = s.strip()

    if len(s) >= 2 and s[1] == ")":
        return s[0]
    return s

def load_fitz_by_relpath(csv_path):
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = (row.get("relative_path") or "").strip()
            fitz = (row.get("fitzpatrick_skin_type") or "").strip()
            if rel:
                mapping[rel] = fitz
    return mapping


def eval_results_for_file(result_path, rel2fitz):

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    total_overall = 0
    correct_overall = 0
    skipped_no_meta = 0
    skipped_no_fitz = 0

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error in file {result_path}, line: {line[:80]}...")
                continue

            image = (item.get("image") or "").strip()
            gt = (item.get("ground_truth") or "").strip()
            pred = (item.get("model_response") or "").strip()

            if not image:
                continue


            fitz = rel2fitz.get(image, None)
            if fitz is None:
                skipped_no_meta += 1
                continue
            if fitz == "":
                skipped_no_fitz += 1
                continue

            total_overall += 1
            stats[fitz]["total"] += 1


            if normalize_option(pred) == normalize_option(gt):
                correct_overall += 1
                stats[fitz]["correct"] += 1

    return {
        "stats": stats,
        "total_overall": total_overall,
        "correct_overall": correct_overall,
        "skipped_no_meta": skipped_no_meta,
        "skipped_no_fitz": skipped_no_fitz,
    }


def print_fairness_report(result_path, result):
    print("\n" + "=" * 80)
    print(f"Result file: {result_path}")
    print("=" * 80)
    print(f"Total samples with Fitzpatrick metadata: {result['total_overall']}")
    if result["total_overall"] > 0:
        overall_acc = result["correct_overall"] / result["total_overall"]
    else:
        overall_acc = 0.0
    print(f"Overall accuracy: {result['correct_overall']} / {result['total_overall']} = {overall_acc:.4f}")
    print(f"Skipped samples (metadata not found): {result['skipped_no_meta']}")
    print(f"Skipped samples (metadata found but Fitzpatrick value is empty): {result['skipped_no_fitz']}")

    print("\nResults by Fitzpatrick skin type:")
    print(f"{'SkinType':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 40)

    acc_list = []
    for fitz in sorted(result["stats"].keys(), key=lambda x: (len(x), x)):
        s = result["stats"][fitz]
        total = s["total"]
        correct = s["correct"]
        acc = correct / total if total > 0 else 0.0
        acc_list.append(acc)
        print(f"{fitz:<10} {correct:>8} {total:>8} {acc:>10.4f}")


    if len(acc_list) > 1:
        max_acc = max(acc_list)
        min_acc = min(acc_list)
        if max_acc > 0:
            FS = 1 - (max_acc - min_acc) / max_acc
        else:
            FS = 0.0
    else:
        FS = 1.0

    print("\nFairness Score (FS): {:.4f}".format(FS))
    print("(FS is in [0,1]; closer to 1 means fairer)")
    print("=" * 80)


def collect_result_files(base_dir):
    collected = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".jsonl"):
                continue

            if name.startswith("ddi_4choices_final_") and name.endswith("_results.jsonl"):
                collected.append(os.path.join(root, name))
    collected.sort()
    return collected


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV does not exist: {CSV_PATH}")
        return

    rel2fitz = load_fitz_by_relpath(CSV_PATH)
    print(f"[INFO] Loaded {len(rel2fitz)} items relative_path -> fitzpatrick_skin_type mappings")

    if not os.path.exists(TASK4_DIR):
        print(f"[ERROR] Result directory does not exist: {TASK4_DIR}")
        return

    result_files = collect_result_files(TASK4_DIR)
    if not result_files:
        print(f"[WARN] In directory {TASK4_DIR} no files matching pattern were found "
              f"'ddi_4choices_final_xxxx_results.jsonl' file")
        return

    print(f"[INFO] Found {len(result_files)} result files:")
    for p in result_files:
        print("  -", p)

    for path in result_files:
        result = eval_results_for_file(path, rel2fitz)
        print_fairness_report(path, result)


if __name__ == "__main__":
    main()
