#!/bin/bash

get_state_config() {
    USER=$(whoami)
    STATE_DIR="$PROJECT_SPACE/training_states"
    ARCHIVE_STATE_DIR="$STATE_DIR/archive"
    STATE_FILE="${USER}_${MODEL_NAME}_NODES_${SLURM_JOB_NUM_NODES}_mbs_${PER_DEVICE_BATCH_SIZE}_gas_${GRADIENT_ACCUMULATION_STEPS}_state.json"
    STATE_FILE_PATH="$STATE_DIR/$STATE_FILE"
    mkdir -p $STATE_DIR $ARCHIVE_STATE_DIR
}

update_state_file() {
    local checkpoint_dir="$1"
    local status="$2"
    local exit_code="${3:-0}"
    local increment_restart="${4:-false}"

    if [ "${MAIN_NODE:-true}" != true ]; then
        return 0
    fi
    
    get_state_config
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local job_id="${SLURM_JOB_ID:-unknown}"
    local job_name="${SLURM_JOB_NAME:-unknown}"
    local num_nodes="${SLURM_JOB_NUM_NODES:-1}"
    local hostname=$(hostname)
    local user=$(whoami)
    local pwd=$(pwd)
    local runtime="unknown"
    runtime=$(squeue -j $SLURM_JOB_ID -h -o "%M" 2>/dev/null || echo "unknown")
    
    # Get current restart count and increment if requested
    local current_count=$(get_restart_count)
    if [ "$increment_restart" = "true" ]; then
        current_count=$((current_count + 1))
    fi

    # Dump to JSON state file
    cat > "$STATE_FILE_PATH" << EOF
{
    "checkpoint_info": {
        "checkpoint_dir": "$checkpoint_dir",
        "model_name": "$MODEL_NAME",
        "per_device_batch_size": $PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
        "exists": $([ -d "$checkpoint_dir" ] && echo "true" || echo "false")
    },
    "job_info": {
        "current_job_id": "$job_id",
        "job_name": "$job_name",
        "user": "$user",
        "hostname": "$hostname",
        "num_nodes": $num_nodes,
        "runtime": "$runtime",
        "status": "$status",
        "timestamp": "$timestamp",
        "exit_code": $exit_code
    },
    "restart_info": {
        "restart_count": $current_count,
        "auto_restart_enabled": true
    }
}
EOF
    echo "============================================================================================="
    echo "State file updated: $STATE_FILE_PATH with job status $status (restart count: $current_count)"
    echo "============================================================================================="
}

get_restart_count() {
    if [ "${MAIN_NODE:-true}" != true ]; then
        echo 1
        return
    fi
    get_state_config
    if [ -f "$STATE_FILE_PATH" ]; then
        python3 -c "
import json
try:
    with open('$STATE_FILE_PATH', 'r') as f:
        data = json.load(f)
    print(data.get('restart_info', {}).get('restart_count', 0))
except:
    print(1)
" 2>/dev/null || echo 1
    else
        echo 1
    fi
}

get_checkpoint_from_state() {
    if [ "${MAIN_NODE:-true}" != true ]; then
        return
    fi
    get_state_config
    if [ -f "$STATE_FILE_PATH" ]; then
        python3 -c "
import json
try:
    with open('$STATE_FILE_PATH', 'r') as f:
        data = json.load(f)
    checkpoint_dir = data.get('checkpoint_info', {}).get('checkpoint_dir', '')
    if checkpoint_dir:
        print(checkpoint_dir)
except:
    print('No checkpoint file found at $STATE_FILE_PATH')
" 2>/dev/null
    fi
}


handle_restart() {
    local exit_code="$1"
    local checkpoint_dir="$2"
    local script_path="$3"
    local max_restarts="${4:-5}"

    if [ "${MAIN_NODE:-true}" != true ]; then
        echo 1
        return
    fi
    
    get_state_config

    echo "============================================================================================="
    # Job was completed: 
    # Exit 0: Job completed (or walltime expiration unless we trap it)
    # Exit 47: Custom code for walltime expiration (with trap in train.sh)
    if [ "$exit_code" -eq 0 ]; then
        echo "Training completed successfully!"
        update_state_file "$checkpoint_dir" "completed" "$exit_code"
        if [ "$max_restarts" -eq 0 ]; then
            echo "Not restarting as max_restarts = 0"
            echo "State file moved to archive $ARCHIVE_STATE_DIR"
            mv "$STATE_FILE_PATH" "$ARCHIVE_STATE_DIR/$STATE_FILE-$SLURM_JOB_ID"
            return 0
        fi
    elif [ "$exit_code" -eq 1 ]; then
        echo "Training interrupted!"
        update_state_file "$checkpoint_dir" "failed" "$exit_code"
        echo "State file moved to archive $ARCHIVE_STATE_DIR"
        if [ "$max_restarts" -eq 0 ]; then
            echo "Not restarting as max_restarts = 0"
            mv "$STATE_FILE_PATH" "$ARCHIVE_STATE_DIR/$STATE_FILE-$SLURM_JOB_ID"
            return 1
        fi
    else
        echo "Training failed or was interrupted (exit code: $exit_code). Manual intervention needed!"
        update_state_file "$checkpoint_dir" "failed" "$exit_code"
        mv "$STATE_FILE_PATH" "$ARCHIVE_STATE_DIR/$STATE_FILE-$SLURM_JOB_ID"
        return 1
    fi

    RESTART_COUNT=$(get_restart_count)
    echo "Current restart count: $RESTART_COUNT, Max allowed: $max_restarts"
    
    if [ "$RESTART_COUNT" -le "$max_restarts" ]; then
        echo "Auto-restarting job $((RESTART_COUNT + 1))/$max_restarts..."
        update_state_file "$checkpoint_dir" "$status" "$exit_code" "true"
        sleep 20 # wait for slurm clean-up previous job
        sbatch "$script_path" --auto-restart $max_restarts
        return 0
    else
        echo "Maximum restart attempts ($max_restarts) reached. Closing down..."
        update_state_file "$checkpoint_dir" "max_restarts_reached" "$exit_code"
        mv "$STATE_FILE_PATH" "$ARCHIVE_STATE_DIR/$STATE_FILE-$SLURM_JOB_ID"
        return 0
    fi
}