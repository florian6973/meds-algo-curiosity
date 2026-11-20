#!/usr/bin/env bash
# Algorithm-owned cluster settings. This file is the only SLURM file expected to need routine edits.
# Curiosity: preprocess_data runs on CPU (consumes no predicates); pretrain and infer_trajectories need a GPU.
# Defaults; each can be overridden from the submit environment (e.g. MEMORY=96G STAGES=pretrain submit.sh), so a
# resource change does not need a commit. Size them per stage by submitting stages separately with STAGES.
PARTITION_CPU="${PARTITION_CPU:-cpu}"
PARTITION_GPU="${PARTITION_GPU:-gpu}"
TIME_CPU="${TIME_CPU:-04:00:00}"
TIME_GPU="${TIME_GPU:-48:00:00}"
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-64G}"
GPUS="${GPUS:-1}"
RUN_ROOT="${RUN_ROOT:-runs}"
PROFILE="${PROFILE:-zero_shot_materialized}"
export PROFILE

# Shared artifacts may only be reused after validating their configuration and input identities.
# Embeddings need training/tuning keys to fit a probe. Inference never reads the label values.
# TASK_PREDICATES_FILE is separate from dataset-level PREDICATES_FILE.
stage_args() {
  if [[ -n "${ALGORITHM_CONFIG_FILE:-}" ]]; then
    printf '%s\n' "config_file=$ALGORITHM_CONFIG_FILE"
  fi
  case "$1" in
    preprocess_data) printf '%s\n' "external_meds_dir=${MEDS_ROOT:?set MEDS_ROOT}" "output_data_dir=$SHARED_DIR/data" ;;
    pretrain) printf '%s\n' "input_data_dir=$SHARED_DIR/data" "output_pretrained_algorithm_dir=$SHARED_DIR/pretrained" ;;
    infer_trajectories) printf '%s\n' "input_data_dir=$SHARED_DIR/data" "input_pretrained_algorithm_dir=$SHARED_DIR/pretrained" "external_index_dir=${INDEX_DIR:?set INDEX_DIR to the prediction index}" "output_trajectories_dir=$RUN_DIR/trajectories" "splits=[${SPLITS:-held_out}]" ;;
    predict_zero_shot_materialized) printf '%s\n' "input_data_dir=$SHARED_DIR/data" "external_meds_dir=${MEDS_ROOT:?set MEDS_ROOT}" "input_trajectories_dir=$RUN_DIR/trajectories" "external_index_dir=${INDEX_DIR:?set INDEX_DIR to the prediction index}" "external_task_criteria_file=${TASK_CRITERIA_FILE:?set TASK_CRITERIA_FILE}" "output_predictions_dir=$RUN_DIR/predictions" "splits=[${SPLITS:-held_out}]" "external_predicates_file=${TASK_PREDICATES_FILE:-${PREDICATES_FILE:?set PREDICATES_FILE}}" ;;
    *) return 2 ;;
  esac
}
