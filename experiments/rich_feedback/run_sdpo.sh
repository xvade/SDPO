#!/bin/bash

# Usage: ./run_sdpo.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
SDPO_PATH="/gscratch/scrubbed/sgvtc/SDPO"
LOGS_PATH="/gscratch/scrubbed/sgvtc/SDPO/xvade/logs"
CKPT_PATH="/gscratch/scrubbed/sgvtc/SDPO/xvade/checkpoints"
SIF_PATH="/gscratch/scrubbed/sgvtc/SDPO/sdpo-gh200.sif"

CONFIG_NAME="sdpo"
BASE_JOB_NAME="rlvr"

DATA_PATHS=(
    "datasets/sciknoweval/chemistry"
)

# Fixed Slurm resources
ACCOUNT="amath"
NODES=1
PARTITION="gpu-l40s"
TIME="12:00:00"
ENV="sdpo"
NTASKS_PER_NODE=1
GPUS_PER_NODE=2
MEM=364G
CPUS_PER_TASK=4

# ACCOUNT="amath"
# NODES=1
# PARTITION="gpu-rtx6k"
# TIME="12:00:00"
# ENV="sdpo"
# NTASKS_PER_NODE=1
# GPUS_PER_NODE=8
# MEM=363G
# CPUS_PER_TASK=40

# ACCOUNT="amath"
# NODES=1
# PARTITION="ckpt"
# TIME="12:00:00"
# ENV="sdpo"
# NTASKS_PER_NODE=1
# GPUS_PER_NODE=8
# MEM=363G
# CPUS_PER_TASK=40
# CONSTRAINT=a40


# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-6)
LAMBDAS=(0.0)
CLIP_ADV_HIGHS=(null)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(1.0)

MODEL_PATHS=(
    # "Qwen/Qwen3-8B"
    "Qwen/Qwen2.5-3B-Instruct"
)

WANDB_KEY="$(tr -d '\r\n' < "$SDPO_PATH/xvade/wandb_apikey.txt")"
# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Define the environment setup and command execution
    # We use the user's home directory dynamically
    local setup_cmds="\
pwd; \
ls; \
export HOME=/tmp/$USER;\
export PIP_CACHE_DIR=/tmp/$USER/pip-cache;\
export PYTHONUSERBASE=/tmp/$USER/pyuserbase;\
export WANDB_API_KEY="$WANDB_KEY";\
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt;\
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt;\
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt;\
pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
pip install -e .; \
pip install --upgrade wandb; \
export PYTHONPATH=.:\$PYTHONPATH; \
"

    local run_cmd="bash ./training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    local apptainer_cmd="bash ./xvade/bin/run_command_in_apptainer.sh $SIF_PATH $SDPO_PATH $LOGS_PATH $CKPT_PATH"

    local inner_cmd="$setup_cmds $run_cmd"

    local apptainer_cmd="bash $SDPO_PATH/xvade/bin/run_command_in_apptainer.sh $SIF_PATH $SDPO_PATH $LOGS_PATH $CKPT_PATH \"$inner_cmd\""

    local wrapped_cmd="srun bash -lc '$apptainer_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        # --environment="$ENV"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="$SDPO_PATH/xvade/output/SDPO/%j.log"
        --error="$SDPO_PATH/xvade/output/SDPO/%j.log"
        --constraint="$CONSTRAINT"
        --wrap="$wrapped_cmd"
    )

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
                            # 1. Construct the experiment name (must be unique)
                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                            EXP_NAME="SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"

                            # 2. Construct the arguments string to pass to the training script
                            # Format: key=value key2=value2 ...
                            
                            OG_ARG_BLOCK="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-rich-feedback \
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
actor_rollout_ref.rollout.val_kwargs.n=4"

                            L40S_ARG_BLOCK="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=vilin97-uw \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=8 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
\
actor_rollout_ref.rollout.max_num_seqs=8"

                            ARGS=$L40S_ARG_BLOCK

                            # 3. Submit
                            submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done

