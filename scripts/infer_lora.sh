#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-dataset_final}"
LORA_MODE="${LORA_MODE:-sft}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

case "${LORA_MODE}" in
    sft)
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-sft}"
        DESCRIPTION_MODULE="src.infer.lora.sft.task1_1_3_1_description"
        DESCRIPTION_MORPH_MODULE="src.infer.lora.sft.task1_2_description_system_prompt"
        MCQ_MODULE="src.infer.lora.sft.task1_3_1_4_2_1_mcq"
        MCQ_25_MODULE="${MCQ_MODULE}"
        HIERARCHICAL_MODULE="src.infer.lora.sft.task2_2_hierarchical"
        COT_MODULE="${DESCRIPTION_MODULE}"
        COT_MORPH_MODULE="src.infer.lora.sft.task3_2_cot_morph_system_prompt"
        TASK2_1_PRESET="${TASK2_1_PRESET:-snu134_4choices}"
        ;;
    rl)
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-rl}"
        DESCRIPTION_MODULE="src.infer.lora.rl.task1_1_1_2_3_1_3_2_description"
        DESCRIPTION_MORPH_MODULE="${DESCRIPTION_MODULE}"
        MCQ_MODULE="src.infer.lora.rl.task1_3_1_4_2_1_mcq"
        MCQ_25_MODULE="${MCQ_MODULE}"
        HIERARCHICAL_MODULE="src.infer.lora.rl.task2_2_hierarchical"
        COT_MODULE="${DESCRIPTION_MODULE}"
        COT_MORPH_MODULE="${DESCRIPTION_MODULE}"
        TASK2_1_PRESET="${TASK2_1_PRESET:-snu134_4choices}"
        ;;
    tta)
        OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-rl_tta_v10.1}"
        DESCRIPTION_MODULE=""
        DESCRIPTION_MORPH_MODULE=""
        MCQ_MODULE="src.infer.lora.tta.task1_3_1_4_2_1_mcq_v10_1_4choices"
        MCQ_25_MODULE="src.infer.lora.tta.task1_3_1_4_2_1_mcq_v10_1_25choices"
        HIERARCHICAL_MODULE="src.infer.lora.rl.task2_2_hierarchical"
        COT_MODULE="src.infer.lora.rl.task1_1_1_2_3_1_3_2_description"
        COT_MORPH_MODULE="${COT_MODULE}"
        TASK2_1_PRESET="${TASK2_1_PRESET:-benchmark}"
        ;;
    *)
        echo "Unknown LORA_MODE='${LORA_MODE}'. Use sft, rl, or tta." >&2
        exit 1
        ;;
esac

export PYTHONPATH="src:${PYTHONPATH:-}"

run_description_module() {
    local module="$1"
    local json_file="$2"
    local json_dir="$3"
    if [[ -z "${module}" ]]; then
        echo "Task is not available for LORA_MODE=${LORA_MODE}." >&2
        exit 1
    fi
    python -m "${module}" \
        --device "${DEVICE}" \
        --model-path "${MODEL_PATH}" \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files "${json_file}" \
        --json-dir "${json_dir}"
}

run_task_1_1() {
    echo "[Task 1.1] description_wo_morph"
    run_description_module \
        "${DESCRIPTION_MODULE}" \
        task1_1_final.jsonl \
        "${DATASET_ROOT}/benchmark/task1/1_1_description_wo_morph"
}

run_task_1_2() {
    echo "[Task 1.2] description_w_morph"
    run_description_module \
        "${DESCRIPTION_MORPH_MODULE}" \
        task1_2_final.jsonl \
        "${DATASET_ROOT}/benchmark/task1/1_2_description_w_morph"
}

run_mcq_file() {
    local label="$1"
    local module="$2"
    local json_file="$3"
    local output_dir="$4"
    echo "${label}"
    python -m "${module}" \
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
        "${MCQ_MODULE}" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt/derm7pt_test_mcq.json" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt"
}

run_task_1_4() {
    run_mcq_file \
        "[Task 1.4] SkinCon MCQ" \
        "${MCQ_MODULE}" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon/skincon_all_mcq.json" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon"
}

run_task_2_1() {
    case "${TASK2_1_PRESET}" in
        snu134_4choices)
            run_mcq_file \
                "[Task 2.1] SNU134 4-choice diagnosis" \
                "${MCQ_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/snu134/task2.1_snu134_4choices.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/snu134"
            ;;
        benchmark)
            run_mcq_file \
                "[Task 2.1] 4-choice diagnosis" \
                "${MCQ_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices"
            run_mcq_file \
                "[Task 2.1] 25-choice diagnosis" \
                "${MCQ_25_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices/task2.1_25choices_gpt_test_2k_non_uniform_sample_final.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices"
            run_mcq_file \
                "[Task 2.1] Derm1M edu subset" \
                "${MCQ_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu/task2.1_derm1m_edu_final.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu"
            run_mcq_file \
                "[Task 2.1] DDI" \
                "${MCQ_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi/task2.1_ddi_4choices.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi"
            run_mcq_file \
                "[Task 2.1] Derm7pt" \
                "${MCQ_MODULE}" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm7pt/task2.1_derm7pt_4choices.json" \
                "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm7pt"
            ;;
        *)
            echo "Unknown TASK2_1_PRESET='${TASK2_1_PRESET}'. Use snu134_4choices or benchmark." >&2
            exit 1
            ;;
    esac
}

run_task_2_2() {
    echo "[Task 2.2] hierarchical diagnosis"
    python -m "${HIERARCHICAL_MODULE}" \
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
        "${MCQ_MODULE}" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality/task2.3_modal_final.json" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality"
}

run_task_3_1() {
    echo "[Task 3.1] CoT + diagnosis"
    run_description_module \
        "${COT_MODULE}" \
        task3_1_final.jsonl \
        "${DATASET_ROOT}/benchmark/task3/3_1"
}

run_task_3_2() {
    echo "[Task 3.2] CoT + morphology + diagnosis"
    run_description_module \
        "${COT_MORPH_MODULE}" \
        task3_2_final.jsonl \
        "${DATASET_ROOT}/benchmark/task3/3_2"
}

run_task_4_1() {
    run_mcq_file \
        "[Task 4.1] DDI fairness" \
        "${MCQ_MODULE}" \
        "${DATASET_ROOT}/benchmark/task4/ddi_4choices_final.json" \
        "${DATASET_ROOT}/benchmark/task4"
}

usage() {
    echo "Usage: MODEL_PATH=... bash $0 {1.1|1.2|1.3|1.4|2.1|2.2|2.3|3.1|3.2|4.1|all}"
    echo "Optional: LORA_MODE=sft|rl|tta DATASET_ROOT=dataset_final DEVICE=cuda:0 NUM_WORKERS=1 OUTPUT_SUFFIX=name TASK2_1_PRESET=snu134_4choices|benchmark"
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
