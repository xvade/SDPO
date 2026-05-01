#!/bin/bash

# Usage: ./run_lean_sdpo.sh [--dry-run]
#
# Prereqs:
#   1. Kimina server must be running as a SLURM job:
#        cd /gscratch/scrubbed/sgvtc/klone-kimina-setup && bash submit_server.sh
#      Wait until $KIMINA_DISCOVERY_DIR/*.addr exists before submitting training.
#   2. Dataset must exist at datasets/lean/minif2f/:
#        python data/make_lean_dataset.py

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

SDPO_PATH="/gscratch/scrubbed/sgvtc/SDPO"
LOGS_PATH="/gscratch/scrubbed/sgvtc/SDPO/xvade/logs"
CKPT_PATH="/gscratch/scrubbed/sgvtc/SDPO/xvade/checkpoints"
SIF_PATH="/gscratch/scrubbed/sgvtc/SDPO/sdpo-gh200.sif"

CONFIG_NAME="sdpo"
BASE_JOB_NAME="rlvr-lean"

DATA_PATHS=(
    "datasets/lean/minif2f"
)

KIMINA_DISCOVERY_DIR="/mmfs1/gscratch/scrubbed/sgvtc/kimina_server_discovery"
KIMINA_CLIENT_SRC="/mmfs1/gscratch/scrubbed/sgvtc/kimina-engine"

# Fixed Slurm resources
ACCOUNT="amath"
NODES=1
PARTITION="gpu-l40s"
TIME="12:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=2
MEM=364G
CPUS_PER_TASK=8
CONSTRAINT=""

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-6)
ALPHAS=(1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

MODEL_PATHS=(
    "AI-MO/Kimina-Prover-Distill-1.7B"
)

WANDB_KEY="$(tr -d '\r\n' < "$SDPO_PATH/xvade/wandb_apikey.txt" 2>/dev/null || true)"

# =============================================================================
# PREFLIGHT CHECK
# =============================================================================

if ! ls "$KIMINA_DISCOVERY_DIR"/*.addr &>/dev/null; then
    echo "WARNING: No Kimina server address file found in $KIMINA_DISCOVERY_DIR"
    echo "         Submit the server first:"
    echo "           cd /gscratch/scrubbed/sgvtc/klone-kimina-setup && bash submit_server.sh"
    echo "         Then wait for the *.addr file to appear before running training."
    if [ "$DRY_RUN" = false ]; then
        echo "Aborting. Use --dry-run to bypass this check."
        exit 1
    fi
fi

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    local setup_cmds="\
pwd; \
ls; \
export HOME=/tmp/$USER;\
export PIP_CACHE_DIR=/tmp/$USER/pip-cache;\
export PYTHONUSERBASE=/tmp/$USER/pyuserbase;\
export WANDB_API_KEY=$WANDB_KEY;\
export KIMINA_DISCOVERY_DIR=$KIMINA_DISCOVERY_DIR;\
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt;\
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt;\
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt;\
pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
pip install $KIMINA_CLIENT_SRC; \
pip install -e .; \
pip install --upgrade wandb; \
export PYTHONPATH=.:\$PYTHONPATH; \
"

    local run_cmd="bash ./training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"
    local apptainer_cmd="bash $SDPO_PATH/xvade/bin/run_command_in_apptainer.sh $SIF_PATH $SDPO_PATH $LOGS_PATH $CKPT_PATH \"$setup_cmds $run_cmd\""
    local wrapped_cmd="srun bash -lc '$apptainer_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="$SDPO_PATH/xvade/output/SDPO/%j.log"
        --error="$SDPO_PATH/xvade/output/SDPO/%j.log"
        --wrap="$wrapped_cmd"
    )

    if [ -n "$CONSTRAINT" ]; then
        sbatch_cmd+=(--constraint="$CONSTRAINT")
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        echo "Submitting job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                            EXP_NAME="LEAN-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"

                            ARG_BLOCK="data.train_batch_size=$TRAIN_BATCH_SIZE \
reward_model.reward_manager=lean \
trainer.group_name=vilin97-uw \
trainer.resume_mode=auto \
trainer.rollout_data_dir=/users/sgvtc/SDPO/xvade/rollouts/$EXP_NAME \
trainer.validation_data_dir=/users/sgvtc/SDPO/xvade/rollouts/${EXP_NAME}_val \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
ray_kwargs.ray_init.num_cpus=8 \
actor_rollout_ref.rollout.max_num_seqs=8"

                            submit_job "$EXP_NAME" "$ARG_BLOCK" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done
