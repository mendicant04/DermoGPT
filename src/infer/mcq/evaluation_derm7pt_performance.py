import json
import sys
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from collections import OrderedDict

# =============================================================================
# Category Definition (7-Point Criteria Mapping)
# =============================================================================
# (This part is the same as before)
CATEGORY_TUPLES = [
    ('pigment_network', '1. Pigment Network (PN)'),
    ('blue_whitish_veil', '2. Blue Whitish Veil (BWV)'),
    ('vascular_structures', '3. Vascular Structures (VS)'),
    ('regression_structures', '7. Regression Structures (RS)'),
    ('dots_and_globules', '6. Dots and Globules (DaG)'),
    ('pigmentation', '4. Pigmentation (PIG)'),
    ('streaks', '5. Streaks (STR)'),
    ('pn', '1. Pigment Network (PN)'),
    ('bwv', '2. Blue Whitish Veil (BWV)'),
    ('vs', '3. Vascular Structures (VS)'),
    ('dag', '6. Dots and Globules (DaG)'),
    ('rs', '7. Regression Structures (RS)'),
    ('pig', '4. Pigmentation (PIG)'),
    ('str', '5. Streaks (STR)'),
]
CATEGORY_NAMES = sorted(list(set([name for _, name in CATEGORY_TUPLES])))

def get_category_from_id(id_string):
    """Maps an ID string to its corresponding category name."""
    id_lower = id_string.lower()
    for key, category_name in CATEGORY_TUPLES:
        if key in id_lower:
            return category_name
    return "Unknown"

# =============================================================================
# Main Evaluation Script (V3 - Map-based Matching)
# =============================================================================

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare LoRA and Baseline model outputs for the derm7pt task."
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
    print(f"Step 1: Loading LoRA data into Map...\n  File: {lora_file_path}")
    combined_data = {}  # {id -> {gt, lora_pred, category}}
    ids_in_lora = set()
    try:
        with open(lora_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    item_id = data.get('id')
                    gt = data.get('ground_truth')
                    lora_pred = data.get('model_response')

                    if not (item_id and gt and isinstance(gt, str) and len(gt) > 0 and
                            lora_pred and isinstance(lora_pred, str) and len(lora_pred) > 0):
                        print(f"  Warning (LoRA File): Skipping line {i + 1}, 'id', 'ground_truth', or 'model_response' is missing/invalid.", file=sys.stderr)
                        continue
                    
                    gt_label = gt[0]
                    lora_pred_label = lora_pred[0]
                    category = get_category_from_id(item_id)
                    
                    if item_id in combined_data:
                        print(f"  Warning (LoRA File): Duplicate ID '{item_id}'. The later entry will be used.", file=sys.stderr)
                    
                    combined_data[item_id] = {
                        'gt': gt_label,
                        'lora_pred': lora_pred_label,
                        'category': category
                    }
                    ids_in_lora.add(item_id)

                except json.JSONDecodeError:
                    print(f"  Warning (LoRA File): Skipping line {i + 1}, JSON format error.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: LoRA file not found: {lora_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading LoRA file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Step 1 Complete: Loaded {len(combined_data)} valid entries from LoRA file.")

    # --- Step 2: Load Baseline (File 2) Data and Merge ---
    print(f"\nStep 2: Loading Baseline data and merging...\n  File: {baseline_file_path}")
    ids_in_baseline = set()
    gt_mismatch_count = 0
    items_matched = 0
    try:
        with open(baseline_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    item_id = data.get('id')
                    base_gt = data.get('ground_truth')
                    base_pred = data.get('model_response')
                    
                    if not (item_id and base_gt and isinstance(base_gt, str) and len(base_gt) > 0 and
                            base_pred and isinstance(base_pred, str) and len(base_pred) > 0):
                        print(f"  Warning (Baseline File): Skipping line {i + 1}, 'id', 'ground_truth', or 'model_response' is missing/invalid.", file=sys.stderr)
                        continue
                    
                    ids_in_baseline.add(item_id)
                    base_pred_label = base_pred[0]
                    base_gt_label = base_gt[0]

                    if item_id in combined_data:
                        items_matched += 1
                        combined_data[item_id]['base_pred'] = base_pred_label
                        
                        # Validate if ground_truth is consistent
                        if combined_data[item_id]['gt'] != base_gt_label:
                            print(f"  Warning (GT Mismatch): ID '{item_id}' has inconsistent ground_truth. "
                                  f"LoRA file: '{combined_data[item_id]['gt']}', Baseline file: '{base_gt_label}'. "
                                  f"Using GT from LoRA file.", file=sys.stderr)
                            gt_mismatch_count += 1
                    else:
                        print(f"  Warning (Baseline File): ID '{item_id}' exists in Baseline file but not in LoRA file. Skipping this entry.", file=sys.stderr)
                
                except json.JSONDecodeError:
                    print(f"  Warning (Baseline File): Skipping line {i + 1}, JSON format error.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Baseline file not found: {baseline_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Baseline file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Step 2 Complete: Matched {items_matched} entries.")
    if gt_mismatch_count > 0:
        print(f"!! Found {gt_mismatch_count} ground_truth mismatches. Evaluation will use GT from the LoRA file.", file=sys.stderr)

    # --- Check ID Differences ---
    lora_only = ids_in_lora - ids_in_baseline
    if lora_only:
        print(f"  Warning: {len(lora_only)} IDs from LoRA file not found in Baseline file. These entries will not be evaluated.", file=sys.stderr)
        # print(f"  (Examples: {list(lora_only)[:5]})") # Uncomment to see examples

    # --- Step 3: Build Evaluation Lists from Merged Map ---
    print("\nStep 3: Organizing data by category for evaluation...")

    # Initialize results dictionaries
    categories_list = CATEGORY_NAMES + ['Overall', 'Unknown']
    lora_results = {name: {'y_true': [], 'y_pred': []} for name in categories_list}
    baseline_results = {name: {'y_true': [], 'y_pred': []} for name in categories_list}

    items_to_evaluate = 0
    for item_id, item in combined_data.items():
        # Only entries present in both files (i.e., 'base_pred' has been added) will be evaluated
        if 'base_pred' not in item:
            continue
        
        items_to_evaluate += 1
        gt = item['gt']
        lora_pred = item['lora_pred']
        base_pred = item['base_pred']
        category = item['category']
        
        # Add to specific category
        lora_results[category]['y_true'].append(gt)
        lora_results[category]['y_pred'].append(lora_pred)
        baseline_results[category]['y_true'].append(gt)
        baseline_results[category]['y_pred'].append(base_pred)
        
        # Add to 'Overall'
        lora_results['Overall']['y_true'].append(gt)
        lora_results['Overall']['y_pred'].append(lora_pred)
        baseline_results['Overall']['y_true'].append(gt)
        baseline_results['Overall']['y_pred'].append(base_pred)

    print(f"Step 3 Complete: Prepared {items_to_evaluate} entries (present in both files) for evaluation.")

    # --- Step 4: Calculate Metrics by Category ---
    print("\n" + "=" * 70)
    print("          Starting Detailed Performance Evaluation by Category")
    print("=" * 70)

    summary_data = []  # For storing the final summary table data

    for category_name in categories_list:
        print(f"\n\n--- Evaluating Category: {category_name} ---")
        
        # y_true for lora and baseline are now guaranteed to be identical
        y_true = lora_results[category_name]['y_true']
        y_pred_lora = lora_results[category_name]['y_pred']
        y_pred_base = baseline_results[category_name]['y_pred']

        # Validation
        if not y_true:
            print("No data for this category, skipping.")
            continue
            
        # Double-check, just in case
        if baseline_results[category_name]['y_true'] != y_true:
            print("!! Critical Error: Internal logic error, y_true lists do not match. Skipping.", file=sys.stderr)
            continue
        
        print(f"(Sample Count: {len(y_true)})")
        labels = sorted(list(set(y_true)))
        
        try:
            # --- LoRA Model Metrics ---
            print("\n[LoRA Model Performance]")
            acc_lora = accuracy_score(y_true, y_pred_lora)
            f1_macro_lora = f1_score(y_true, y_pred_lora, average='macro', zero_division=0)
            report_lora = classification_report(y_true, y_pred_lora, labels=labels, zero_division=0)
            
            print(f"  Accuracy: {acc_lora:.4f}")
            print(f"  F1 Score (Macro): {f1_macro_lora:.4f}")
            print("  Classification Report:")
            print(report_lora)

            # --- Baseline Model Metrics ---
            print("\n[Baseline Model Performance]")
            acc_baseline = accuracy_score(y_true, y_pred_base)
            f1_macro_baseline = f1_score(y_true, y_pred_base, average='macro', zero_division=0)
            report_baseline = classification_report(y_true, y_pred_base, labels=labels, zero_division=0)
            
            print(f"  Accuracy: {acc_baseline:.4f}")
            print(f"  F1 Score (Macro): {f1_macro_baseline:.4f}")
            print("  Classification Report:")
            print(report_baseline)
            
            # --- Store Summary Data ---
            summary_data.append({
                'Category': category_name,
                'Samples': len(y_true),
                'LoRA Acc': f"{acc_lora:.4f}",
                'LoRA F1 (Macro)': f"{f1_macro_lora:.4f}",
                'Baseline Acc': f"{acc_baseline:.4f}",
                'Baseline F1 (Macro)': f"{f1_macro_baseline:.4f}",
            })
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    # =============================================================================
    # Final Summary
    # =============================================================================

    print("\n\n" + "=" * 80)
    print("                                Performance Summary Comparison")
    print("=" * 80)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Reorder rows based on category
        category_order = CATEGORY_NAMES + ['Overall', 'Unknown']
        summary_df['Category'] = pd.Categorical(summary_df['Category'], categories=category_order, ordered=True)
        summary_df = summary_df.sort_values('Category')
        
        print(summary_df.to_string(index=False))
    else:
        print("Failed to generate summary data for any category.")

    print("\nScript execution finished.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)