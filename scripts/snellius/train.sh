#!/bin/bash

set -ex

# Print requested GPUs and nodes to .out
export MAIN_NODE=false
if [[ "$SLURM_NODEID" == "0" ]]; then
    [[ "$TRAIN_MODE" == "debug" ]] && set -x
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
    echo "Number of nodes: $SLURM_NNODES"
    export MAIN_NODE=true
fi


# ============================================================================
# STARTING CONFIGS
# ============================================================================
WORKING_DIR=$PWD
STATE_DIR="$PROJECT_SPACE/training_states"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/training_state_utils.sh"

if [[ "$AUTO_RESTART" -gt 0 ]]; then
    echo "Auto-restart enabled with max restarts: $AUTO_RESTART"
else
    echo "Auto-restart disabled (max restarts: $AUTO_RESTART)"
fi

# ============================================================================
# ENVIRONMENT
# ============================================================================
module purge
module load 2024 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0

export VENV_PATH=$PWD/"venv"
source $VENV_PATH/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# NCCL flight recorder for stack traces on timeout (zero-cost until failure)
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

# ============================================================================
# TRAINING CONFIG
# ============================================================================
# if CONFIG_PATH is not yet set, default value
export CONFIG_PATH=${CONFIG_PATH:-$PWD/config/gpt-nl-26B-mid-train.yaml}

export MODEL_NAME=gpt-nl-$MODEL_SIZE

# CAREFUL this checkpoint_dir will be overwritten if auto_restart is enabled and a running training state is found. See below!
export CHECKPOINT_DIR="$PROJECT_SPACE/$MODEL_NAME-nodes-$SLURM_NNODES-mbs-$PER_DEVICE_BATCH_SIZE-gas-$GRADIENT_ACCUMULATION_STEPS-$SLURM_JOB_ID"
export OLMO_SHARED_FS=1

# ============================================================================
# DISTRIBUTED
# ============================================================================
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591

# TMPDIR must be on a SHARED path (same on all nodes) because olmo-core uses it
# as work_dir for global data indices that rank 0 writes and all ranks read.
# Cache dirs (inductor/triton/xdg) are per-node to avoid rm -rf race.
SHARED_BASE="/scratch-shared/$USER/cache/$SLURM_JOBID"
NODE_CACHE="${SHARED_BASE}/node_${SLURM_NODEID}"
export TMPDIR="${SHARED_BASE}/tmp"

# Unique per-node cache directories (not shared)
export TORCHINDUCTOR_CACHE_DIR="${NODE_CACHE}/inductor_${SLURM_PROCID}"
export TRITON_CACHE_DIR="${NODE_CACHE}/triton_${SLURM_PROCID}"
export XDG_CACHE_HOME="${NODE_CACHE}/xdg_${SLURM_PROCID}"

# Clean only THIS node's cache dirs (not TMPDIR — that's shared)
rm -rf "$NODE_CACHE"
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $XDG_CACHE_HOME/torch/kernels

# ============================================================================
# WEIGHTS & BIASES
# ============================================================================
source scripts/snellius/.env
# default value if not set previously
PROJECT_SUFFIX=${PROJECT_SUFFIX:-"-train"}
IS_CONTEXT_EXTENSION=${IS_CONTEXT_EXTENSION:-false}
export WANDB_PROJECT="GPT-NL-${MODEL_SIZE}${PROJECT_SUFFIX}"
if [ "$IS_CONTEXT_EXTENSION" = "true" ]; then
    PYTHON_SCRIPT="train_context.py"
else
    PYTHON_SCRIPT="train.py"
fi
export WANDB_MODE=online




# ============================================================================
# CHECKPOINT DIRECTORY LOGIC
# ============================================================================

# Check if resume from existing checkpoint
EXISTING_CHECKPOINT=$(get_checkpoint_from_state)
if [ -n "$EXISTING_CHECKPOINT" ] && [ -d "$EXISTING_CHECKPOINT" ]; then
    export CHECKPOINT_DIR="$EXISTING_CHECKPOINT"
    echo "Resuming from existing checkpoint: $CHECKPOINT_DIR"
    update_state_file "$CHECKPOINT_DIR" "resuming"
else
    echo "No existing checkpoint found. Starting from: $CHECKPOINT_DIR"
    update_state_file "$CHECKPOINT_DIR" "starting"
fi

# Find latest checkpoint. OLMo-core may not always create the 'latest' symlink
LATEST_CHECKPOINT=$(ls "$CHECKPOINT_DIR" | grep '^step[0-9]\+$' | sed 's/step//' | sort -n | tail -n1)
if [ -n "$LATEST_CHECKPOINT" ] && [ -z "$LOAD_PATH" ]; then
    LATEST_CHECKPOINT="step$LATEST_CHECKPOINT"
    export LOAD_PATH="$CHECKPOINT_DIR/$LATEST_CHECKPOINT"
    echo "Latest checkpoint directory: $CHECKPOINT_DIR"
fi

if [ "$MAIN_NODE" = "true" ]; then
    update_state_file "$CHECKPOINT_DIR" "running"
fi


# ============================================================================
# PROFILING AND HARDWARE CHECK
# ============================================================================
# Set value but can be overwritten with sbatch ... --profile --gpu-health-check


export NSYS_RUN=""
if [[ -n $PROFILING ]]; then
   echo "nsys trace file will be found in the traces folder upon termination"
   mkdir -p $WORKING_DIR/traces
   export NSYS_RUN="nsys profile --stats=true --trace=cuda -o traces/trace_${SLURM_JOBID} --cuda-memory-usage=true" # --trace=cuda,cudnn,cublas,osrt,nvtx 
fi

if [[ -n $GPU_HEALTH_CHECK ]]; then
   echo "GPU health check is enabled"

   # Initialize container input and outputs
   CONTAINER=/INSERT-GPT-NL-DATALOADER-DIR/gpu_health_check/gpu_health_check.sif
   GPU_HEALTH_DIR=${WORKING_DIR}/gpu_health_results
   mkdir -p $GPU_HEALTH_DIR
   rm $GPU_HEALTH_DIR/* -rf

   # Run the GPU health check across all requested nodes
   apptainer run --nv $CONTAINER run --output $GPU_HEALTH_DIR
   # Analyze the results and return as variable
   GPU_CHECK=$(apptainer run --nv $CONTAINER analyze --output $GPU_HEALTH_DIR)

   if [[ "$GPU_CHECK" == *"All GPUs are operating as expected."* ]]; then
    echo "GPU check passed. Continuing launching training..."
   else
      echo "GPU check failed. Message: $CHECK_OUTPUT"
      exit 1
   fi
fi

export JOB_START=$(date +%s)
if [ "$MAIN_NODE" = "true" ]; then
    LIMIT_STR=$(scontrol show job "$SLURM_JOB_ID" | grep -o "TimeLimit=[^ ]*" | cut -d'=' -f2)

    # Convert SLURM time format to seconds
    # SLURM format can be: "5-00:00:00" (days-hours:minutes:seconds) or "01:30:00" (hh:mm:ss)
    if [[ "$LIMIT_STR" == *-* ]]; then
        IFS='-:' read -r days hours minutes seconds <<<"$LIMIT_STR"
        days=${days:-0}
        hours=${hours:-0}
        minutes=${minutes:-0}
        seconds=${seconds:-0}
        TIME_LIMIT_SECONDS=$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
    else
        IFS=':' read -r hours minutes seconds <<<"$LIMIT_STR"
        days=0
        hours=${hours:-0}
        minutes=${minutes:-0}
        seconds=${seconds:-0}
        TIME_LIMIT_SECONDS=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
    fi

    export TIME_LIMIT_SECONDS
else
    TIME_LIMIT_SECONDS=0
fi
echo $TIME_LIMIT_SECONDS

LAUNCH_JOBFILE=${LAUNCH_JOBFILE:-"launch_train_debug.job"}

graceful_exit() {
    if [ "${MAIN_NODE:-true}" != true ]; then
        return 0
    fi
    source "$SCRIPT_DIR/training_state_utils.sh"
    echo "$(date): Received SIGTERM - stopping training..."
    NOW=$(date +%s)
    ELAPSED=$((NOW - JOB_START))
    REMAINING_SECONDS=$((${TIME_LIMIT_SECONDS:-0} - ELAPSED))
    if [ "$REMAINING_SECONDS" -le 120 ]; then
        echo "$(date): SIGTERM occurred within 60s of walltime – assuming walltime preemption"
        EXIT_CODE=0
    else
        echo "$(date): SIGTERM occurred earlier – assuming manual cancellation"
        EXIT_CODE=1
    fi
    echo "$(date): Calling handle_restart with exit code $EXIT_CODE"
    # Call handle_restart and exit
    handle_restart $EXIT_CODE "$CHECKPOINT_DIR" "${SCRIPT_DIR}/$LAUNCH_JOBFILE" $AUTO_RESTART
    
    echo "$(date): Exiting with code $EXIT_CODE"
    exit $EXIT_CODE
}
[[ "$TRAIN_MODE" == "prod" ]] &&  trap graceful_exit TERM
 
# ============================================================================
# TORCHLAUNCHER
# ============================================================================
export TORCHLAUNCHER="torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --rdzv_id=$SLURM_JOBID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
    scripts/$PYTHON_SCRIPT $CONFIG_PATH"

if [[ "$TRAIN_MODE" == "prod" ]]; then
    eval "$TORCHLAUNCHER" &
    wait $!
else
    eval "$TORCHLAUNCHER"
fi

EXIT_CODE=$?
handle_restart $EXIT_CODE "$CHECKPOINT_DIR" "${SCRIPT_DIR}/launch_train.job" $AUTO_RESTART