#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-dataset_final}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3-vl-8b}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

case "${MODEL_FAMILY}" in
    lingshu-7b)
        MODULE_PREFIX="${MODULE_PREFIX:-src.infer.baseline.lingshu-7b}"
        MODEL_BASE="${MODEL_BASE:-Qwen/lingshu-7b-Instruct}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-lingshu-7b}"
        ;;
    lingshu-32b)
        MODULE_PREFIX="${MODULE_PREFIX:-src.infer.baseline.lingshu-7b}"
        MODEL_BASE="${MODEL_BASE:-Qwen/lingshu-7b-Instruct}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-lingshu-32b}"
        ;;
    qwen3-vl-8b)
        MODULE_PREFIX="${MODULE_PREFIX:-src.infer.baseline.qwen3-vl-8b}"
        MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-8B-Instruct}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-qwen3-vl-8b-baseline}"
        ;;
    skinvl-mm)
        MODULE_PREFIX="${MODULE_PREFIX:-src.infer.baseline.skinvl-mm}"
        MODEL_BASE="${MODEL_BASE:-Qwen/skinvl-mm-Instruct}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-skinvl-mm}"
        : "${SKINVL_MM_ROOT:?Set SKINVL_MM_ROOT for SkinVL-MM}"
        export PYTHONPATH="${SKINVL_MM_ROOT}:src:${PYTHONPATH:-}"
        ;;
    custom)
        : "${MODULE_PREFIX:?Set MODULE_PREFIX for MODEL_FAMILY=custom}"
        : "${MODEL_BASE:?Set MODEL_BASE for MODEL_FAMILY=custom}"
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-custom}"
        ;;
    *)
        echo "Unknown MODEL_FAMILY='${MODEL_FAMILY}'." >&2
        echo "Use lingshu-7b, lingshu-32b, qwen3-vl-8b, skinvl-mm, or custom." >&2
        exit 1
        ;;
esac

if [[ "${MODEL_FAMILY}" != "skinvl-mm" ]]; then
    export PYTHONPATH="src:${PYTHONPATH:-}"
fi

run_task_1_1() {
    echo "[Task 1.1] description_wo_morph"
    python -m "${MODULE_PREFIX}.task1_1_description" \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --json-files task1_1_final.jsonl \
        --json-dir "${DATASET_ROOT}/benchmark/task1/1_1_description_wo_morph"
}

run_task_1_2() {
    echo "[Task 1.2] description_w_morph"
    python -m "${MODULE_PREFIX}.task1_2_description_system_prompt" \
        --model-path "${MODEL_PATH}" \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --json-files task1_2_final.jsonl \
        --json-dir "${DATASET_ROOT}/benchmark/task1/1_2_description_w_morph"
}

run_mcq_file() {
    local label="$1"
    local json_file="$2"
    local output_dir="$3"
    echo "${label}"
    python -m "${MODULE_PREFIX}.task1_3_1_4_2_1_mcq_prompt" \
        --model-path "${MODEL_PATH}" \
        --loaded-model-name "${OUTPUT_SUFFIX}" \
        --model-base "${MODEL_BASE}" \
        --device "${DEVICE}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files "${json_file}" \
        --output-dir "${output_dir}" \
        --num-workers "${NUM_WORKERS}"
}

run_task_1_3() {
    run_mcq_file \
        "[Task 1.3] Derm7pt MCQ" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt/derm7pt_test_mcq.json" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt"
}

run_task_1_4() {
    run_mcq_file \
        "[Task 1.4] SkinCon MCQ" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon/skincon_all_mcq.json" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon"
}

run_task_2_1() {
    run_mcq_file \
        "[Task 2.1] 4-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices"

    run_mcq_file \
        "[Task 2.1] 25-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices/task2.1_25choices_gpt_test_2k_non_uniform_sample_final.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices"

    run_mcq_file \
        "[Task 2.1] Derm1M edu subset" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu/task2.1_derm1m_edu_final.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu"
}

run_task_2_2() {
    echo "[Task 2.2] hierarchical diagnosis"
    python -m "${MODULE_PREFIX}.task2_2_hierarchical" \
        --model-path "${MODEL_PATH}" \
        --loaded-model-name "${OUTPUT_SUFFIX}" \
        --model-base "${MODEL_BASE}" \
        --device "${DEVICE}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-file "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final.json" \
        --output-dir "${DATASET_ROOT}/benchmark/task2/2_2_hierarchical" \
        --num-workers "${NUM_WORKERS}"
}

run_task_2_3() {
    run_mcq_file \
        "[Task 2.3] image modality" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality/task2.3_modal_final.json" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality"
}

run_task_3_1() {
    echo "[Task 3.1] CoT + diagnosis"
    python -m "${MODULE_PREFIX}.task3_1_cot_system_prompt" \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files task3_1_final.jsonl \
        --json-dir "${DATASET_ROOT}/benchmark/task3/3_1"
}

run_task_3_2() {
    echo "[Task 3.2] CoT + morphology + diagnosis"
    python -m "${MODULE_PREFIX}.task3_2_cot_morph_system_prompt" \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files task3_2_final.jsonl \
        --json-dir "${DATASET_ROOT}/benchmark/task3/3_2"
}

run_task_4_1() {
    run_mcq_file \
        "[Task 4.1] DDI fairness" \
        "${DATASET_ROOT}/benchmark/task4/ddi_4choices_final.json" \
        "${DATASET_ROOT}/benchmark/task4"
}

usage() {
    echo "Usage: MODEL_PATH=... bash $0 {1.1|1.2|1.3|1.4|2.1|2.2|2.3|3.1|3.2|4.1|all}"
    echo "Optional: MODEL_FAMILY=lingshu-7b|lingshu-32b|qwen3-vl-8b|skinvl-mm|custom DATASET_ROOT=dataset_final DEVICE=cuda:0 NUM_WORKERS=1 OUTPUT_SUFFIX=name"
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 1
fi

for task in "$@"; do
    case "${task}" in
        1.1) run_task_1_1 ;;
        1.2) run_task_1_2 ;;
        1.3) run_task_1_3 ;;
        1.4) run_task_1_4 ;;
        2.1) run_task_2_1 ;;
        2.2) run_task_2_2 ;;
        2.3) run_task_2_3 ;;
        3.1) run_task_3_1 ;;
        3.2) run_task_3_2 ;;
        4.1) run_task_4_1 ;;
        all)
            run_task_1_1
            run_task_1_2
            run_task_1_3
            run_task_1_4
            run_task_2_1
            run_task_2_2
            run_task_2_3
            run_task_3_1
            run_task_3_2
            run_task_4_1
            ;;
        *)
            echo "Unknown task '${task}'." >&2
            usage
            exit 1
            ;;
    esac
done
