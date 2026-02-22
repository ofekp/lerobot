#!/bin/bash
# =============================================================================
# process_libero.sh — Automated training and evaluation of GROOT on LIBERO tasks
# =============================================================================
#
# Prerequisites:
#   - Datasets have been replayed using libero/process_libero.sh
#     and are available at /data_host/{libero_suite}_replay/{camera_name}/{task_name}_demo/
#   - Run inside the depth_vla Docker container with 4 GPUs available
#   - Working directory: /workspace/lerobot
#
# Strategy:
#   - Process 2 tasks at a time, each in RGB and RGBD mode (= 4 jobs on 4 GPUs)
#   - GPU 0: task i,   RGB
#   - GPU 1: task i,   RGBD
#   - GPU 2: task i+1, RGB
#   - GPU 3: task i+1, RGBD
#   - After each training batch, eval all 4 checkpoints (same GPU assignment)
#
# Usage:
#   bash scripts/process_libero.sh
#
# =============================================================================

set -euo pipefail

# ---- Load .env file if present (same directory as this script) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a  # auto-export all sourced variables
    source "${SCRIPT_DIR}/.env"
    set +a
    echo "[INFO] Loaded ${SCRIPT_DIR}/.env"
fi

# ---- Experiment name (used in output paths, wandb job names, email) ----
EXP_NAME="korean"

# ---- Telegram notifications ----
# After each batch of 2 tasks finishes eval, post a summary via Telegram bot.
# Set up:
#   1. Message @BotFather on Telegram → /newbot → copy the token
#   2. Message your bot, then visit https://api.telegram.org/bot<TOKEN>/getUpdates
#      to find your chat_id
#   3. Paste both values below (or set env vars)
NOTIFY_ENABLED=true
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

cd /workspace/lerobot
export PYTHONPATH=/workspace/lerobot:
export WANDB_API_KEY="e8bdac9000c6a752b23ae5002e1baf18e67a87fc"

# ---- Sharding (for multi-machine runs) ----
# Tasks are split round-robin by index: task i goes to shard (i % NUM_SHARDS).
# Auto-detected from hostname; override with SHARD_ID= and NUM_SHARDS= env vars.
NUM_SHARDS=${NUM_SHARDS:-2}
if [ -z "${SHARD_ID:-}" ]; then
    if [[ "${HOST_HOSTNAME:-}" == *newai-gpu07* ]]; then
        SHARD_ID=0
    elif [[ "${HOST_HOSTNAME:-}" == *newai-gpu02* ]]; then
        SHARD_ID=1
    else
        echo "[WARN] Unknown HOST_HOSTNAME='${HOST_HOSTNAME:-}' — defaulting to SHARD_ID=0 (run all tasks)."
        echo "[WARN] Set SHARD_ID=0 or SHARD_ID=1 explicitly, or set NUM_SHARDS=1 to run everything."
        SHARD_ID=0
        NUM_SHARDS=1
    fi
fi
echo "[INFO] Sharding: shard $((SHARD_ID+1)) of ${NUM_SHARDS} (host: ${HOST_HOSTNAME:-unknown})"

# ---- Configuration ----
CAMERA_NAMES=("robot0_eye_in_hand" "frontview")
LIBERO_SUITES=("libero_object" "libero_goal")

NUM_STEPS=20000
BATCH_SIZE=32
SAVE_FREQ=5000
LOG_FREQ=100
SEED=142

EVAL_N_EPISODES=50
EVAL_BATCH_SIZE=10

# Depth settings (for RGBD mode) — identity transform, depth stays in meters
DEPTH_WEIGHT_INIT="rgb_average"
DEPTH_SCALE=1.0
DEPTH_MEAN=0.0
DEPTH_STD=1.0

# ---- Helper: get the eval camera name (LIBERO appends _image) ----
get_eval_camera_name() {
    echo "${1}_image"
}

# ---- Helper: camera_name_mapping for RGB eval ----
get_camera_name_mapping_rgb() {
    echo "{\"${1}_image\": \"image\"}"
}

# ---- Helper: camera_name_mapping for RGBD eval ----
get_camera_name_mapping_rgbd() {
    echo "{\"${1}_image\": \"image\", \"${1}_depth\": \"image.depth\"}"
}

# ---- Helper: look up task_id by matching folder name against the LIBERO benchmark ----
# Folder names end with _demo (from hdf5 filenames); benchmark names don't.
get_task_id() {
    local suite="$1"
    local task_name="$2"  # e.g. "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo"
    local bddl_name="${task_name%_demo}"  # strip _demo suffix
    # tail -1: libero prints info messages to stdout; we only want the last line (the numeric id)
    python3 -c "
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()['${suite}']()
names = suite.get_task_names()
try:
    print(names.index('${bddl_name}'))
except ValueError:
    print(-1)
" 2>&1 | tail -1
}

# ---- Helper: check if a task (train+eval) is already done ----
# A task is done if eval_info.json exists and contains ["overall"]["pc_success"]
is_task_done() {
    local suite="$1"
    local camera="$2"
    local task_name="$3"
    local mode="$4"  # "rgb" or "rgbd"

    local eval_json="./output_${EXP_NAME}/${suite}/${camera}/${task_name}/${mode}/eval/eval_info.json"
    if [ ! -f "${eval_json}" ]; then
        return 1  # not done
    fi
    # Check that the file has overall.pc_success
    python3 -c "
import json, sys
try:
    with open('${eval_json}') as f:
        data = json.load(f)
    val = data['overall']['pc_success']
    sys.exit(0)  # done
except (KeyError, json.JSONDecodeError, FileNotFoundError):
    sys.exit(1)  # not done
"
}

# ---- Helper: check if training is complete (final checkpoint exists) ----
# Training is done if the checkpoint at NUM_STEPS exists.
is_train_done() {
    local suite="$1"
    local camera="$2"
    local task_name="$3"
    local mode="$4"

    # Format step number with leading zeros (6 digits) to match checkpoint folder name
    local step_dir
    step_dir=$(printf "%06d" "${NUM_STEPS}")
    local checkpoint="./output_${EXP_NAME}/${suite}/${camera}/${task_name}/${mode}/checkpoints/${step_dir}/pretrained_model/model.safetensors"
    [ -f "${checkpoint}" ]
}

# ---- Helper: read pc_success from eval_info.json ----
get_pc_success() {
    local suite="$1"
    local camera="$2"
    local task_name="$3"
    local mode="$4"

    local eval_json="./output_${EXP_NAME}/${suite}/${camera}/${task_name}/${mode}/eval/eval_info.json"
    python3 -c "
import json
try:
    with open('${eval_json}') as f:
        data = json.load(f)
    print(f\"{data['overall']['pc_success']:.1f}\")
except Exception:
    print('N/A')
"
}

# ---- Helper: send a simple text message via Telegram ----
send_telegram_message() {
    local text="$1"

    if [ "${NOTIFY_ENABLED}" != "true" ] || [ -z "${TELEGRAM_BOT_TOKEN}" ] || [ -z "${TELEGRAM_CHAT_ID}" ]; then
        return 0
    fi

    TG_TEXT="${text}" \
    TG_TOKEN="${TELEGRAM_BOT_TOKEN}" \
    TG_CHAT="${TELEGRAM_CHAT_ID}" \
    python3 << 'PYEOF'
import json, urllib.request, os

token = os.environ["TG_TOKEN"]
chat_id = os.environ["TG_CHAT"]
text = os.environ["TG_TEXT"]

payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
data = json.dumps(payload).encode("utf-8")
url = f"https://api.telegram.org/bot{token}/sendMessage"
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req) as resp:
        print(f"[TG] Message sent (HTTP {resp.status})")
except Exception as e:
    print(f"[TG] Failed to send: {e}")
PYEOF
}

# ---- Helper: send Telegram notification after a batch ----
send_batch_notification() {
    local suite="$1"
    local camera="$2"
    shift 2
    local task_names=("$@")

    if [ "${NOTIFY_ENABLED}" != "true" ]; then
        return 0
    fi
    if [ -z "${TELEGRAM_BOT_TOKEN}" ] || [ -z "${TELEGRAM_CHAT_ID}" ]; then
        echo "[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping notification."
        return 0
    fi

    # Build results text lines
    local results_lines=""
    for tn in "${task_names[@]}"; do
        local rgb_score rgbd_score
        rgb_score=$(get_pc_success "${suite}" "${camera}" "${tn}" "rgb")
        rgbd_score=$(get_pc_success "${suite}" "${camera}" "${tn}" "rgbd")
        results_lines+="  • <b>${tn}</b>\n    rgb: ${rgb_score}  |  rgbd: ${rgbd_score}\n"
    done

    TG_RESULTS="${results_lines}" \
    TG_EXP="${EXP_NAME}" \
    TG_SUITE="${suite}" \
    TG_CAMERA="${camera}" \
    TG_HOST="${HOST_HOSTNAME:-unknown}" \
    TG_SHARD="$((SHARD_ID+1)) of ${NUM_SHARDS}" \
    TG_TOKEN="${TELEGRAM_BOT_TOKEN}" \
    TG_CHAT="${TELEGRAM_CHAT_ID}" \
    python3 << 'PYEOF'
import json, urllib.request, os

exp = os.environ['TG_EXP']
suite = os.environ['TG_SUITE']
camera = os.environ['TG_CAMERA']
host = os.environ['TG_HOST']
shard = os.environ['TG_SHARD']
results = os.environ['TG_RESULTS']
token = os.environ['TG_TOKEN']
chat_id = os.environ['TG_CHAT']

text = (
    f"\U0001f4ca <b>[{exp}] Eval complete</b>\n"
    f"\n"
    f"  \U00002699 <b>Suite:</b>  {suite}\n"
    f"  \U0001f4f7 <b>Camera:</b>  {camera}\n"
    f"  \U0001f5a5 <b>Host:</b>  {host}\n"
    f"  \U0001f4e6 <b>Shard:</b>  {shard}\n"
    f"\n"
    f"<b>Results:</b>\n"
    f"{results}"
)

payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
data = json.dumps(payload).encode("utf-8")
url = f"https://api.telegram.org/bot{token}/sendMessage"
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req) as resp:
        print(f"[TG] Batch notification sent (HTTP {resp.status})")
except Exception as e:
    print(f"[TG] Failed to send: {e}")
PYEOF
}

# ---- Train function ----
run_train() {
    local gpu="$1"
    local mode="$2"       # "rgb" or "rgbd"
    local suite="$3"      # e.g. "libero_object"
    local camera="$4"     # e.g. "frontview"
    local task_name="$5"  # e.g. "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo"
    local dataset_path="$6"

    local output_dir="./output_${EXP_NAME}/${suite}/${camera}/${task_name}/${mode}"
    local job_name="${EXP_NAME}_${suite}_${camera}_${task_name}_${mode}"

    echo "[TRAIN] GPU=${gpu} | ${mode} | ${suite}/${camera}/${task_name}"

    local depth_args=""
    if [ "$mode" = "rgbd" ]; then
        depth_args="--policy.use_depth=true --policy.depth_weight_init=${DEPTH_WEIGHT_INIT} --policy.depth_scale=${DEPTH_SCALE} --policy.depth_mean=${DEPTH_MEAN} --policy.depth_std=${DEPTH_STD}"
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python3 -m lerobot.scripts.lerobot_train \
        --output_dir="${output_dir}" \
        --save_checkpoint=true \
        --batch_size=${BATCH_SIZE} \
        --steps=${NUM_STEPS} \
        --save_freq=${SAVE_FREQ} \
        --log_freq=${LOG_FREQ} \
        --policy.push_to_hub=false \
        --policy.type=groot \
        --policy.tune_diffusion_model=false \
        --dataset.repo_id="${dataset_path}" \
        --wandb.enable=true \
        --wandb.disable_artifact=true \
        --job_name="${job_name}" \
        --seed=${SEED} \
        ${depth_args}
}

# ---- Eval function ----
run_eval() {
    local gpu="$1"
    local mode="$2"
    local suite="$3"
    local camera="$4"
    local task_name="$5"
    local task_id="$6"

    local output_dir="./output_${EXP_NAME}/${suite}/${camera}/${task_name}/${mode}"
    local checkpoint_dir="${output_dir}/checkpoints/last/pretrained_model/"
    local eval_output_dir="${output_dir}/eval/"
    local eval_camera
    eval_camera=$(get_eval_camera_name "${camera}")

    echo "[EVAL]  GPU=${gpu} | ${mode} | ${suite}/${camera}/${task_name} (task_id=${task_id})"

    if [ ! -d "${checkpoint_dir}" ]; then
        echo "[EVAL]  WARNING: Checkpoint not found at ${checkpoint_dir}, skipping."
        return 1
    fi

    local depth_args=""
    local camera_mapping
    if [ "$mode" = "rgbd" ]; then
        camera_mapping=$(get_camera_name_mapping_rgbd "${camera}")
        depth_args="--env.use_depth=true --policy.use_depth=true --policy.depth_scale=${DEPTH_SCALE} --policy.depth_mean=${DEPTH_MEAN} --policy.depth_std=${DEPTH_STD}"
    else
        camera_mapping=$(get_camera_name_mapping_rgb "${camera}")
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python3 -m lerobot.scripts.lerobot_eval \
        --env.type=libero \
        --env.task="${suite}" \
        --env.gym_kwargs="{\"task_ids\": [${task_id}]}" \
        --env.camera_name="${eval_camera}" \
        --env.camera_name_mapping="${camera_mapping}" \
        --policy.type=groot \
        --policy.pretrained_path="${checkpoint_dir}" \
        --output_dir="${eval_output_dir}" \
        --eval.batch_size=${EVAL_BATCH_SIZE} \
        --eval.n_episodes=${EVAL_N_EPISODES} \
        --policy.device=cuda \
        ${depth_args}
}

# =============================================================================
# Pre-flight: check for incomplete (partially trained) output directories
# Dirs with training done but no eval are OK — they'll be eval-only.
# Only flag dirs where training itself is incomplete (started but not finished).
# =============================================================================
echo ""
echo "[INFO] Checking for incomplete output directories..."
INCOMPLETE_DIRS=()
EVAL_ONLY_DIRS=()
for _camera_name in "${CAMERA_NAMES[@]}"; do
    for _libero_suite in "${LIBERO_SUITES[@]}"; do
        _DATA_ROOT="/data_host/${_libero_suite}_replay/${_camera_name}"
        [ ! -d "${_DATA_ROOT}" ] && continue
        while IFS= read -r _task_dir; do
            _task_name=$(basename "$_task_dir")
            for _mode in rgb rgbd; do
                _output_dir="./output_${EXP_NAME}/${_libero_suite}/${_camera_name}/${_task_name}/${_mode}"
                if [ -d "${_output_dir}" ]; then
                    if is_task_done "${_libero_suite}" "${_camera_name}" "${_task_name}" "${_mode}"; then
                        : # Fully done — skip
                    elif is_train_done "${_libero_suite}" "${_camera_name}" "${_task_name}" "${_mode}"; then
                        EVAL_ONLY_DIRS+=("${_output_dir}")
                    else
                        INCOMPLETE_DIRS+=("${_output_dir}")
                    fi
                fi
            done
        done < <(find "${_DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
    done
done

if [ ${#EVAL_ONLY_DIRS[@]} -gt 0 ]; then
    echo "[INFO] Found ${#EVAL_ONLY_DIRS[@]} directories with training done but eval pending:"
    for _dir in "${EVAL_ONLY_DIRS[@]}"; do
        echo "  ↳ ${_dir}"
    done
    echo "[INFO] These will be evaluated without re-training."
fi

if [ ${#INCOMPLETE_DIRS[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "[ERROR] Found ${#INCOMPLETE_DIRS[@]} incomplete output directories!"
    echo "[ERROR] These have partial training (checkpoint at step ${NUM_STEPS} NOT found)."
    echo "[ERROR] Please delete them and re-run:"
    echo ""
    for _dir in "${INCOMPLETE_DIRS[@]}"; do
        echo "  rm -rf ${_dir}"
    done
    echo ""
    echo "Or delete all at once:"
    echo "  rm -rf ${INCOMPLETE_DIRS[*]}"
    echo "============================================================"
    exit 1
fi
echo "[INFO] No incomplete directories found. Proceeding."

# =============================================================================
# Main loop
# =============================================================================
for camera_name in "${CAMERA_NAMES[@]}"; do
    for libero_suite in "${LIBERO_SUITES[@]}"; do
        DATA_ROOT="/data_host/${libero_suite}_replay/${camera_name}"

        if [ ! -d "${DATA_ROOT}" ]; then
            echo "[WARN] Data directory not found: ${DATA_ROOT}, skipping."
            continue
        fi

        # Collect task folders (explicitly sorted for consistent ordering across machines)
        TASK_DIRS=()
        while IFS= read -r task_dir; do
            TASK_DIRS+=("$task_dir")
        done < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)

        # Filter tasks for this shard (round-robin by index)
        SHARD_TASK_DIRS=()
        for (( t=0; t<${#TASK_DIRS[@]}; t++ )); do
            if (( t % NUM_SHARDS == SHARD_ID )); then
                SHARD_TASK_DIRS+=("${TASK_DIRS[$t]}")
            fi
        done

        echo ""
        echo "============================================================"
        echo "Suite: ${libero_suite} | Camera: ${camera_name} | Tasks: ${#SHARD_TASK_DIRS[@]}/${#TASK_DIRS[@]} (shard $((SHARD_ID+1)) of ${NUM_SHARDS})"
        echo "============================================================"

        # Process tasks in pairs (2 tasks × 2 modes = 4 GPUs)
        for (( i=0; i<${#SHARD_TASK_DIRS[@]}; i+=2 )); do
            PIDS=()

            # Resolve task info for both tasks in this batch
            declare -a BATCH_TASKS=()
            declare -a BATCH_IDS=()
            for j in 0 1; do
                idx=$((i + j))
                if [ $idx -ge ${#SHARD_TASK_DIRS[@]} ]; then
                    continue
                fi

                task_dir="${SHARD_TASK_DIRS[$idx]}"
                task_name=$(basename "$task_dir")
                task_id=$(get_task_id "${libero_suite}" "${task_name}")

                if [ "$task_id" = "-1" ]; then
                    echo "[ERROR] Could not find task_id for ${task_name} in ${libero_suite}, skipping."
                    continue
                fi

                BATCH_TASKS+=("${task_name}")
                BATCH_IDS+=("${task_id}")
                echo "[INFO] Batch task: ${task_name} → task_id=${task_id}"
            done

            # --- Check which tasks need training/eval ---
            ALL_DONE=true
            for (( j=0; j<${#BATCH_TASKS[@]}; j++ )); do
                task_name="${BATCH_TASKS[$j]}"
                for mode in rgb rgbd; do
                    if is_task_done "${libero_suite}" "${camera_name}" "${task_name}" "${mode}"; then
                        echo "[SKIP] Fully done (train+eval): ${task_name}/${mode}"
                    elif is_train_done "${libero_suite}" "${camera_name}" "${task_name}" "${mode}"; then
                        echo "[INFO] Training done, eval needed: ${task_name}/${mode}"
                        ALL_DONE=false
                    else
                        echo "[INFO] Training + eval needed: ${task_name}/${mode}"
                        ALL_DONE=false
                    fi
                done
            done

            if [ "$ALL_DONE" = true ]; then
                echo "[SKIP] All tasks in this batch are fully done. Skipping."
                unset BATCH_TASKS BATCH_IDS
                echo ""
                continue
            fi

            # --- Launch training (4 parallel jobs, skip if already trained) ---
            echo "[INFO] Starting training batch (tasks ${i} and $((i+1)))..."
            send_telegram_message "$(printf '\xF0\x9F\x9A\x80 <b>[%s] Training started</b>\n\n  Suite: %s\n  Camera: %s\n  Host: %s\n  Shard: %s of %s\n\n  Tasks:\n  %s' "${EXP_NAME}" "${libero_suite}" "${camera_name}" "${HOST_HOSTNAME:-unknown}" "$((SHARD_ID+1))" "${NUM_SHARDS}" "$(printf '• %s\n  ' "${BATCH_TASKS[@]}")" )"
            for (( j=0; j<${#BATCH_TASKS[@]}; j++ )); do
                task_name="${BATCH_TASKS[$j]}"
                dataset_path="${DATA_ROOT}/${task_name}"
                gpu_rgb=$((j * 2))
                gpu_rgbd=$((j * 2 + 1))

                if ! is_train_done "${libero_suite}" "${camera_name}" "${task_name}" "rgb"; then
                    run_train ${gpu_rgb}  "rgb"  "${libero_suite}" "${camera_name}" "${task_name}" "${dataset_path}" &
                    PIDS+=($!)
                fi
                if ! is_train_done "${libero_suite}" "${camera_name}" "${task_name}" "rgbd"; then
                    run_train ${gpu_rgbd} "rgbd" "${libero_suite}" "${camera_name}" "${task_name}" "${dataset_path}" &
                    PIDS+=($!)
                fi
            done

            # Wait for all training jobs in this batch
            if [ ${#PIDS[@]} -gt 0 ]; then
                for pid in "${PIDS[@]}"; do
                    wait "$pid" || echo "[WARN] Training PID $pid exited with non-zero status"
                done
            fi
            echo "[INFO] Training batch done."

            # --- Launch eval (4 parallel jobs, same GPU assignment) ---
            # Eval runs for any task that is trained but not yet evaluated
            echo "[INFO] Starting eval batch..."
            EVAL_PIDS=()
            for (( j=0; j<${#BATCH_TASKS[@]}; j++ )); do
                task_name="${BATCH_TASKS[$j]}"
                task_id="${BATCH_IDS[$j]}"
                gpu_rgb=$((j * 2))
                gpu_rgbd=$((j * 2 + 1))

                if ! is_task_done "${libero_suite}" "${camera_name}" "${task_name}" "rgb" && \
                   is_train_done "${libero_suite}" "${camera_name}" "${task_name}" "rgb"; then
                    run_eval ${gpu_rgb}  "rgb"  "${libero_suite}" "${camera_name}" "${task_name}" "${task_id}" &
                    EVAL_PIDS+=($!)
                fi
                if ! is_task_done "${libero_suite}" "${camera_name}" "${task_name}" "rgbd" && \
                   is_train_done "${libero_suite}" "${camera_name}" "${task_name}" "rgbd"; then
                    run_eval ${gpu_rgbd} "rgbd" "${libero_suite}" "${camera_name}" "${task_name}" "${task_id}" &
                    EVAL_PIDS+=($!)
                fi
            done

            if [ ${#EVAL_PIDS[@]} -gt 0 ]; then
                for pid in "${EVAL_PIDS[@]}"; do
                    wait "$pid" || echo "[WARN] Eval PID $pid exited with non-zero status"
                done
            fi
            echo "[INFO] Eval batch done."

            # --- Send Teams notification for this batch ---
            send_batch_notification "${libero_suite}" "${camera_name}" "${BATCH_TASKS[@]}"

            # Clean up batch arrays
            unset BATCH_TASKS BATCH_IDS
            echo ""
        done

        echo "[INFO] All tasks done for ${libero_suite} / ${camera_name}."
    done
done

echo ""
echo "============================================================"
echo "All suites and cameras processed."
echo "============================================================"