# compare_task1_results.py
# -*- coding: utf-8 -*-

import os, json, csv, argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def read_per_item(p: Path):
    items=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): items.append(json.loads(line))

    m={}
    for obj in items:
        _id=obj["id"]; m[_id]=float(obj.get("final_overall", obj.get("text_aggregate",{}).get("overall",0.0)))
    return m, items

def find_per_item_in_dir(d: Path) -> Path:
    cand = d / "per_item.jsonl"
    if cand.exists(): return cand

    raise FileNotFoundError(f"per_item.jsonl not found in {d}")

def parse_inputs(dirs: List[str], inputs: List[str]) -> Dict[str, Path]:
    out={}
    if dirs:
        for d in dirs:
            alias = Path(d).name
            out[alias] = find_per_item_in_dir(Path(d))
    if inputs:
        for spec in inputs:

            if "=" not in spec: raise ValueError(f"--inputs item must be alias=path, got: {spec}")
            alias, path = spec.split("=",1)
            out[alias] = Path(path)
    if len(out)<2:
        raise ValueError("Need at least two models to compare")
    return out

def write_csv(rows, path: Path):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out_dir", default="./eval_task1/compare")
    args = ap.parse_args()

    mapping = parse_inputs(args.dirs, args.inputs)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)


    model_scores = {}   # alias -> {id: final_overall}
    model_items  = {}   # alias -> list[records]
    for alias, peritem_path in mapping.items():
        m, items = read_per_item(peritem_path)
        model_scores[alias]=m; model_items[alias]=items


    common_ids = None
    for alias, m in model_scores.items():
        ids=set(m.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids) if common_ids else []
    if not common_ids:
        print("[WARN] No overlapping ids among models; abort leaderboard/pairwise.")
        return


    leaderboard=[]
    for alias, m in model_scores.items():
        vals=[m[_id] for _id in common_ids]
        mean = round(sum(vals)/len(vals), 4) if vals else 0.0
        leaderboard.append({"alias": alias, "n_ids": len(vals), "mean_final_overall": mean})
    leaderboard.sort(key=lambda x: x["mean_final_overall"], reverse=True)
    write_csv(leaderboard, out_root / "leaderboard.csv")

    # Pairwise A/B
    aliases=list(model_scores.keys())
    for i in range(len(aliases)):
        for j in range(i+1, len(aliases)):
            A, B = aliases[i], aliases[j]
            pair_dir = out_root / "pairwise" / f"{A}__vs__{B}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            rows=[]; wins=loses=ties=0; diffs=[]
            for _id in common_ids:
                oa, ob = model_scores[A][_id], model_scores[B][_id]
                diff=ob-oa
                if abs(diff) < 1e-6: out="tie"; ties+=1
                elif diff>0: out="B_win"; wins+=1
                else: out="A_win"; loses+=1
                rows.append({"id":_id, "A_final":oa, "B_final":ob, "B_minus_A":round(diff,2), "outcome":out})
                diffs.append(diff)
            write_csv(rows, pair_dir / "head_to_head.csv")
            macro={
                "n_overlap": len(common_ids),
                "B_wins": wins, "A_wins": loses, "ties": ties,
                "B_minus_A_mean": round(sum(diffs)/len(diffs), 4) if diffs else 0.0,
                "B_minus_A_median": round(sorted(diffs)[len(diffs)//2], 4) if diffs else 0.0
            }
            (pair_dir / "macro_compare.json").write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Compare -> {out_root}")

if __name__ == "__main__":
    main()
