# compare_task1_2_results.py
# -*- coding: utf-8 -*-

import os, json, csv, argparse
from pathlib import Path
from typing import Dict, List

def read_per_item(p: Path):
    items=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): items.append(json.loads(line))

    m_final={}
    m_text={}
    m_morph={}
    m_sem={}
    m_auto={}
    m_pen={}
    for obj in items:
        _id = obj["id"]
        m_final[_id] = float(obj.get("final_overall", 0.0))
        m_text[_id]  = float(obj.get("text_aggregate",{}).get("overall", 0.0))
        m_morph[_id] = float(obj.get("morph_score_0_100", 0.0))
        m_sem[_id]   = float(obj.get("morph_semantic",{}).get("score_semantic", 0.0))
        m_auto[_id]  = float(obj.get("morph_auto_score_0_1", 0.0))
        m_pen[_id]   = float(obj.get("cross_consistency",{}).get("penalty", 0.0))
    return {"final":m_final, "text":m_text, "morph":m_morph, "sem":m_sem, "auto":m_auto, "pen":m_pen}

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
            if "=" not in spec: raise ValueError(f"--inputs needs alias=path, got: {spec}")
            alias, path = spec.split("=",1)
            out[alias] = Path(path)
    if len(out) < 2:
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
    ap.add_argument("--out_dir", default="./eval_task1_2/compare")
    args = ap.parse_args()

    mapping = parse_inputs(args.dirs, args.inputs)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)


    models = {}
    for alias, peritem_path in mapping.items():
        models[alias] = read_per_item(peritem_path)


    common_ids = None
    for alias, packs in models.items():
        ids = set(packs["final"].keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids) if common_ids else []
    if not common_ids:
        print("[WARN] No overlapping ids among models."); return

    # Leaderboard by final_overall
    leaderboard=[]
    components=[]
    for alias, packs in models.items():
        finals = [packs["final"][i] for i in common_ids]
        texts  = [packs["text"][i]  for i in common_ids]
        morphs = [packs["morph"][i] for i in common_ids]
        sems   = [packs["sem"][i]   for i in common_ids]
        autos  = [packs["auto"][i]  for i in common_ids]
        pens   = [packs["pen"][i]   for i in common_ids]
        def mean(x): return round(sum(x)/len(x),4) if x else 0.0
        leaderboard.append({"alias":alias, "n_ids":len(common_ids), "mean_final_overall":mean(finals)})
        components.append({
            "alias":alias, "n_ids":len(common_ids),
            "mean_text_overall":mean(texts),
            "mean_morph_score_0_100":mean(morphs),
            "mean_morph_sem_0_1":mean(sems),
            "mean_morph_auto_0_1":mean(autos),
            "mean_cross_penalty_0_1":mean(pens),
        })
    leaderboard.sort(key=lambda r: r["mean_final_overall"], reverse=True)
    write_csv(leaderboard, out_root / "leaderboard.csv")
    write_csv(components,  out_root / "components.csv")

    # Pairwise
    aliases=list(models.keys())
    for i in range(len(aliases)):
        for j in range(i+1, len(aliases)):
            A, B = aliases[i], aliases[j]
            pair_dir = out_root / "pairwise" / f"{A}__vs__{B}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            rows=[]; wins=loses=ties=0; diffs=[]
            for _id in common_ids:
                oa, ob = models[A]["final"][_id], models[B]["final"][_id]
                d = ob - oa
                if abs(d) < 1e-6: out="tie"; ties+=1
                elif d>0: out="B_win"; wins+=1
                else: out="A_win"; loses+=1
                rows.append({"id":_id, "A_final":oa, "B_final":ob, "B_minus_A":round(d,2), "outcome":out})
                diffs.append(d)
            write_csv(rows, pair_dir / "head_to_head.csv")
            macro={
                "n_overlap": len(common_ids),
                "B_wins": wins, "A_wins": loses, "ties": ties,
                "B_minus_A_mean": round(sum(diffs)/len(diffs),4) if diffs else 0.0,
                "B_minus_A_median": round(sorted(diffs)[len(diffs)//2],4) if diffs else 0.0
            }
            (pair_dir / "macro_compare.json").write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Compare -> {out_root}")

if __name__ == "__main__":
    main()
