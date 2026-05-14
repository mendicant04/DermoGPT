#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-dataset_final}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
DATASET_VARIANT="${DATASET_VARIANT:-mini}"

: "${BASE_URL:?Set BASE_URL}"
: "${API_KEY:?Set API_KEY}"
: "${API_MODEL:?Set API_MODEL}"

if [[ "${DATASET_VARIANT}" == "mini" ]]; then
    DEFAULT_SUFFIX="_mini"
elif [[ "${DATASET_VARIANT}" == "full" ]]; then
    DEFAULT_SUFFIX=""
else
    echo "Invalid DATASET_VARIANT='${DATASET_VARIANT}'. Use 'mini' or 'full'." >&2
    exit 1
fi

TEXT_SUFFIX="${TEXT_SUFFIX:-${DEFAULT_SUFFIX}}"
MCQ_SUFFIX="${MCQ_SUFFIX:-${DEFAULT_SUFFIX}}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-$(printf '%s' "${API_MODEL}" | tr '/:' '__' | tr -cs 'A-Za-z0-9._-' '-')}"

export PYTHONPATH="src:${PYTHONPATH:-}"

api_common_args=(
    --base_url "${BASE_URL}"
    --api_key "${API_KEY}"
    --model "${API_MODEL}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
)

run_task_1_1() {
    echo "[Task 1.1] description_wo_morph"
    python -m src.infer.baseline.api.task1_1_description \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --json-files "task1_1_final${TEXT_SUFFIX}.jsonl" \
        --json-dir "${DATASET_ROOT}/benchmark/task1/1_1_description_wo_morph" \
        "${api_common_args[@]}"
}

run_task_1_2() {
    echo "[Task 1.2] description_w_morph"
    python -m src.infer.baseline.api.task1_2_description_system_prompt \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --json-files "task1_2_final${TEXT_SUFFIX}.jsonl" \
        --json-dir "${DATASET_ROOT}/benchmark/task1/1_2_description_w_morph" \
        "${api_common_args[@]}"
}

run_mcq_file() {
    local label="$1"
    local json_file="$2"
    local output_dir="$3"
    echo "${label}"
    python -m src.infer.baseline.api.task1_3_1_4_2_1_mcq_prompt \
        --loaded-model-name "${OUTPUT_SUFFIX}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files "${json_file}" \
        --output-dir "${output_dir}" \
        --num-workers "${NUM_WORKERS}" \
        "${api_common_args[@]}"
}

run_optional_mcq_file() {
    local label="$1"
    local json_file="$2"
    local output_dir="$3"
    if [[ -f "${json_file}" ]]; then
        run_mcq_file "${label}" "${json_file}" "${output_dir}"
    else
        echo "[SKIP] ${label}: ${json_file} does not exist."
    fi
}

run_task_1_3() {
    run_mcq_file \
        "[Task 1.3] Derm7pt MCQ" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt/derm7pt_test_mcq${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task1/1_3_mcq_derm7pt"
}

run_task_1_4() {
    run_mcq_file \
        "[Task 1.4] SkinCon MCQ" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon/skincon_all_mcq${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task1/1_4_mcq_skincon"
}

run_task_2_1() {
    run_mcq_file \
        "[Task 2.1] 4-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices"

    run_mcq_file \
        "[Task 2.1] 25-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices/task2.1_25choices_gpt_test_2k_non_uniform_sample_final${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/25_choices"

    run_mcq_file \
        "[Task 2.1] Derm1M edu subset" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu/task2.1_derm1m_edu_final${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm1m_edu"
}

run_task_2_1_extra() {
    run_optional_mcq_file \
        "[Task 2.1 extra] Derm7pt 4-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm7pt/task2.1_derm7pt_4choices.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/derm7pt"

    run_optional_mcq_file \
        "[Task 2.1 extra] DDI 4-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi/task2.1_ddi_4choices.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi"

    run_optional_mcq_file \
        "[Task 2.1 extra] DDI all diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi/task2.1_ddi_all.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/ddi"

    run_optional_mcq_file \
        "[Task 2.1 extra] SNU134 4-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/snu134/task2.1_snu134_4choices.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/snu134"
}

run_task_2_1_cqzyy() {
    run_mcq_file \
        "[Task 2.1] CQZYY 3-choice diagnosis" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/cqzyy/task2.1_cqzyy_3choices.json" \
        "${DATASET_ROOT}/benchmark/task2/2_1_mcq/cqzyy"
}

run_task_2_2() {
    echo "[Task 2.2] hierarchical diagnosis"
    python -m src.infer.baseline.api.task2_2_hierarchical \
        --loaded-model-name "${OUTPUT_SUFFIX}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-file "${DATASET_ROOT}/benchmark/task2/2_1_mcq/4_choices/task2.1_test_2k_non_uniform_sample_final${MCQ_SUFFIX}.json" \
        --output-dir "${DATASET_ROOT}/benchmark/task2/2_2_hierarchical" \
        --num-workers "${NUM_WORKERS}" \
        "${api_common_args[@]}"
}

run_task_2_3() {
    run_mcq_file \
        "[Task 2.3] image modality" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality/task2.3_modal_final${MCQ_SUFFIX}.json" \
        "${DATASET_ROOT}/benchmark/task2/2_3_modality"
}

run_task_3_1() {
    echo "[Task 3.1] CoT + diagnosis"
    python -m src.infer.baseline.api.task3_1_cot_system_prompt \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files "task3_1_final${TEXT_SUFFIX}.jsonl" \
        --json-dir "${DATASET_ROOT}/benchmark/task3/3_1" \
        "${api_common_args[@]}"
}

run_task_3_2() {
    echo "[Task 3.2] CoT + morphology + diagnosis"
    python -m src.infer.baseline.api.task3_2_cot_morph_system_prompt \
        --output-suffix "${OUTPUT_SUFFIX}" \
        --num-workers "${NUM_WORKERS}" \
        --image-base-path "${DATASET_ROOT}" \
        --json-files "task3_2_final${TEXT_SUFFIX}.jsonl" \
        --json-dir "${DATASET_ROOT}/benchmark/task3/3_2" \
        "${api_common_args[@]}"
}

run_task_4_1() {
    run_mcq_file \
        "[Task 4.1] DDI fairness" \
        "${DATASET_ROOT}/benchmark/task4/ddi_4choices_final.json" \
        "${DATASET_ROOT}/benchmark/task4"
}

usage() {
    echo "Usage: BASE_URL=... API_KEY=... API_MODEL=... bash $0 {1.1|1.2|1.3|1.4|2.1|2.1-extra|2.1-cqzyy|2.2|2.3|3.1|3.2|4.1|all}"
    echo "Optional: DATASET_ROOT=dataset_final DATASET_VARIANT=mini|full OUTPUT_SUFFIX=name NUM_WORKERS=1 MAX_NEW_TOKENS=8192"
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
        2.1-extra) run_task_2_1_extra ;;
        2.1-cqzyy) run_task_2_1_cqzyy ;;
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
        all-extra)
            run_task_1_1
            run_task_1_2
            run_task_1_3
            run_task_1_4
            run_task_2_1
            run_task_2_1_extra
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
