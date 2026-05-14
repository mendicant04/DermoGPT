import json
import sys
import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import defaultdict

# =============================================================================
# Label Parsing Function (for Grouping)
# =============================================================================

def extract_label(text):
    """
    Extracts "Label Text" as the category from "X) Label Text" format.
    """
    if not text or not isinstance(text, str):
        return None
    
    split_point = text.find(') ')
    
    if split_point != -1:
        category = text[split_point + 2:].strip()
        return category if category else None
    else:
        # Fallback or error if format is unexpected
        return None

# =============================================================================
# Main Evaluation Script (SkinCon - Acc + Macro-Avg-Acc)
# =============================================================================

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare LoRA and Baseline model outputs for the SkinCon task."
    )
    parser.add_argument(
        '--lora_file',
        type=str,
        required=True,
        help="Path to the .jsonl file containing LoRA model results."
    )
    parser.add_argument(
        '--baseline_file',
        type=str,
        required=True,
        help="Path to the .jsonl file containing Baseline model results."
    )
    return parser.parse_args()

def main(args):
    """Main function to run the evaluation."""
    
    # --- File Paths (from args) ---
    lora_file_path = args.lora_file
    baseline_file_path = args.baseline_file

    # --- Step 1: Load LoRA (File 1) Data into Map ---
    print(f"SkinCon Evaluation - Step 1: Loading LoRA data into Map...\n  File: {lora_file_path}")
    combined_data = {}  # {id -> {gt_char, lora_char, category}}
    ids_in_lora = set()
    try:
        with open(lora_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    item_id = data.get('id')
                    gt_raw = data.get('ground_truth')
                    lora_raw = data.get('model_response')

                    if not (item_id and gt_raw and isinstance(gt_raw, str) and len(gt_raw) > 0 and
                            lora_raw and isinstance(lora_raw, str) and len(lora_raw) > 0):
                        print(f"  Warning (LoRA File): Skipping line {i+1}, 'id', 'ground_truth', or 'model_response' is missing/invalid.", file=sys.stderr)
                        continue
                    
                    category = extract_label(gt_raw)
                    gt_char = gt_raw[0]
                    lora_char = lora_raw[0]
                    
                    if category is None:
                        print(f"  Warning (LoRA File): Skipping line {i+1}, cannot extract category from 'ground_truth': {gt_raw}", file=sys.stderr)
                        continue

                    combined_data[item_id] = {
                        'gt_char': gt_char,
                        'lora_char': lora_char,
                        'category': category
                    }
                    ids_in_lora.add(item_id)
                except json.JSONDecodeError:
                    print(f"  Warning (LoRA File): Skipping line {i+1}, JSON format error.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: LoRA file not found: {lora_file_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Step 1 Complete: Loaded {len(combined_data)} valid entries from LoRA file.")


    # --- Step 2: Load Baseline (File 2) Data and Merge ---
    print(f"\nSkinCon Evaluation - Step 2: Loading Baseline data and merging...\n  File: {baseline_file_path}")
    ids_in_baseline = set()
    gt_mismatch_count = 0
    items_matched = 0
    try:
        with open(baseline_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    item_id = data.get('id')
                    base_gt_raw = data.get('ground_truth')
                    base_raw = data.get('model_response')
                    
                    if not (item_id and base_gt_raw and isinstance(base_gt_raw, str) and len(base_gt_raw) > 0 and
                            base_raw and isinstance(base_raw, str) and len(base_raw) > 0):
                        print(f"  Warning (Baseline File): Skipping line {i+1}, 'id', 'ground_truth', or 'model_response' is missing/invalid.", file=sys.stderr)
                        continue
                    
                    base_gt_char = base_gt_raw[0]
                    base_char = base_raw[0]
                    ids_in_baseline.add(item_id)

                    if item_id in combined_data:
                        items_matched += 1
                        combined_data[item_id]['base_char'] = base_char
                        if combined_data[item_id]['gt_char'] != base_gt_char:
                            print(f"  Warning (GT Char Mismatch): ID '{item_id}' has inconsistent first char in GT.", file=sys.stderr)
                            gt_mismatch_count += 1
                    else:
                        # ID in baseline but not in LoRA map (already handled by LoRA-only check later)
                        pass
                except json.JSONDecodeError:
                    print(f"  Warning (Baseline File): Skipping line {i+1}, JSON format error.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Baseline file not found: {baseline_file_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Step 2 Complete: Matched {items_matched} entries.")


    # --- Step 3: Organize Evaluation Lists by Category ---
    print("\nSkinCon Evaluation - Step 3: Organizing data by category...")
    results_by_cat = defaultdict(lambda: {'y_true': [], 'y_pred_lora': [], 'y_pred_base': []})
    all_categories = set()
    items_to_evaluate = 0

    for item_id, item in combined_data.items():
        if 'base_char' not in item:
            # Item was in LoRA file but not baseline file
            continue
        
        items_to_evaluate += 1
        cat = item['category']
        gt = item['gt_char']
        all_categories.add(cat)
        
        results_by_cat[cat]['y_true'].append(gt)
        results_by_cat[cat]['y_pred_lora'].append(item['lora_char'])
        results_by_cat[cat]['y_pred_base'].append(item['base_char'])
        
        # Also add to 'Overall' category
        results_by_cat['Overall']['y_true'].append(gt)
        results_by_cat['Overall']['y_pred_lora'].append(item['lora_char'])
        results_by_cat['Overall']['y_pred_base'].append(item['base_char'])

    print(f"Step 3 Complete: Prepared {items_to_evaluate} entries, belonging to {len(all_categories)} categories.")

    # --- Step 4: Calculate Accuracy by Category ---
    if items_to_evaluate == 0:
        print("\nError: No matched entries available for evaluation. Please check file paths and IDs.")
        sys.exit(1)

    print("\n" + "="*70)
    print("          Starting Accuracy Evaluation by Category (by-point)")
    print("="*70)

    summary_data = []
    # Lists for calculating macro-average
    lora_category_accs = []
    baseline_category_accs = []

    categories_to_evaluate = sorted(list(all_categories)) + ['Overall']

    for category_name in categories_to_evaluate:
        print(f"\n\n--- Evaluating Category (by-point): {category_name} ---")
        
        cat_data = results_by_cat[category_name]
        y_true = cat_data['y_true']
        y_pred_lora = cat_data['y_pred_lora']
        y_pred_base = cat_data['y_pred_base']
        
        if not y_true:
            print("No data for this category.")
            continue
            
        print(f"(Sample Count: {len(y_true)})")
        
        try:
            # --- LoRA Model Accuracy ---
            acc_lora = accuracy_score(y_true, y_pred_lora)
            print(f"  [LoRA Model]     Accuracy: {acc_lora:.4f}")

            # --- Baseline Model Accuracy ---
            acc_baseline = accuracy_score(y_true, y_pred_base)
            print(f"  [Baseline Model] Accuracy: {acc_baseline:.4f}")
            
            # --- Store summary data ---
            summary_data.append({
                'Category': category_name,
                'Samples': len(y_true),
                'LoRA Acc': f"{acc_lora:.4f}",
                'Baseline Acc': f"{acc_baseline:.4f}",
            })
            
            # --- Store values for macro-average calculation ---
            if category_name != 'Overall':
                lora_category_accs.append(acc_lora)
                baseline_category_accs.append(acc_baseline)
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    # =============================================================================
    # Final Summary
    # =============================================================================
    print("\n\n" + "="*80)
    print("                Accuracy Performance Summary Comparison (SkinCon - by-point category)")
    print("="*80)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df['Category'] = pd.Categorical(summary_df['Category'], categories=categories_to_evaluate, ordered=True)
        summary_df = summary_df.sort_values('Category')
        
        # --- New: Calculate and add macro-average row ---
        if len(lora_category_accs) > 0:
            macro_avg_lora = sum(lora_category_accs) / len(lora_category_accs)
            macro_avg_baseline = sum(baseline_category_accs) / len(baseline_category_accs)
            
            macro_avg_row = pd.DataFrame([{
                'Category': 'Macro-Average (by-point)',
                'Samples': f"{len(lora_category_accs)} categories",
                'LoRA Acc': f"{macro_avg_lora:.4f}",
                'Baseline Acc': f"{macro_avg_baseline:.4f}"
            }])
            
            # Add Macro-Average row before "Overall" row
            overall_row = summary_df[summary_df['Category'] == 'Overall']
            summary_df = summary_df[summary_df['Category'] != 'Overall']
            summary_df = pd.concat([summary_df, macro_avg_row, overall_row], ignore_index=True)

        print(summary_df.to_string(index=False))
    else:
        print("Failed to generate summary data for any category.")

    print("\nScript execution finished.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)