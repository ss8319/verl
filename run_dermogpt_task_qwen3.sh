#!/bin/bash
#SBATCH --job-name=verl_dermogpt
#SBATCH --account=ub62
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=768G
#SBATCH --partition=fit
#SBATCH --time=24:00:00
#SBATCH --output=ssl4rl_%j.out
#SBATCH --error=ssl4rl_%j.err
#SBATCH --exclude=m3u009

# --- Environment Setup ---
# Automatically detect number of GPUs from SLURM
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    N_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    # Parse count from e.g. "0,1,2,3"
    N_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
    # Default to 1 if not in SLURM
    N_GPUS=1
fi
echo "Detected $N_GPUS GPUs allocated for this job."
# Usage: sbatch run_dermogpt_task.sh [TASK] [MODEL_PATH] [DATA_LIMIT]
TASK=${1:-rotation}
MODEL_PATH=${2:-"Qwen/Qwen3-VL-8B-Instruct"}
DATA_LIMIT=${3:-0.25}
RUN_ID=$SLURM_JOB_ID
MODEL_NAME=$(basename "$MODEL_PATH")

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# --- Environment Key Check ---
echo "--- Environment Key Check ---"
MISSING_KEYS=0
if [ -z "$WANDB_API_KEY" ]; then echo "WANDB_API_KEY: NOT SET"; MISSING_KEYS=1; else echo "WANDB_API_KEY: SET"; fi
if [ -z "$HF_TOKEN" ]; then echo "HF_TOKEN: NOT SET"; MISSING_KEYS=1; else echo "HF_TOKEN: SET"; fi

if [ $MISSING_KEYS -eq 1 ]; then
    echo "ERROR: Critical environment keys are missing. Please check your .env file."
    exit 1
fi
echo "----------------------------"

# Ensure ALL cache and config directories are on scratch to avoid home quota issues
export SCRATCH_DIR="/fs04/scratch2/ub62/ssim0070"
export HF_HOME="${SCRATCH_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_DIR}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/.triton"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/.cache/wandb"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export VLLM_USAGE_SOURCE=offline  # Disable vLLM usage reporting to avoid writing to home
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

# Physical path on scratch for container access
WORK_DIR="/fs04/scratch2/ub62/ssim0070/verl"
DATASET_DIR="/fs04/scratch2/ub62/ssim0070/SSL4RL/our_datasets/dermogpt_v3/${TASK}"
SAVE_DIR="models_v3/verl_dermogpt_v3_${TASK}_${MODEL_NAME}_${RUN_ID}"
APPTAINER_IMG="/fs04/scratch2/ub62/ssim0070/verl_vllm017.latest.sif"

# Rollout backend selection
ROLLOUT_BACKEND="vllm"

# Hyperparameters (Optimized for GRPO + LoRA)
# Note: TRAIN_BATCH_SIZE should be >= total samples in dataset.
# If using a very small subset (e.g., 8 samples), set this to 8.
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BSZ_PER_GPU=32
ROLLOUT_N=5  # Number of rollouts per prompt
# Context lengths: For 224x224 images with Qwen3-VL (patch=16, merge=2), each image = 49 tokens
# Typical prompt: ~100 text tokens + 49 image tokens = ~150 tokens
# Response: ~100-200 tokens for <think>...</think><answer>X</answer>
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=1024
VAL_BEFORE_TRAIN=True

# --- Apptainer Command Construction ---
cd "${WORK_DIR}" || exit 1

# Automatically ensure the datasets symlink exists
if [ ! -L "datasets" ] && [ ! -e "datasets" ]; then
    echo "Creating symlink to datasets folder..."
    ln -s /fs04/scratch2/ub62/ssim0070/SSL4RL/datasets datasets
fi

# Export PYTHONPATH to include current verl folder and native-matched Flash Attention
export PYTHONPATH="/fs04/scratch2/ub62/ssim0070/python_pkgs/vllm_017:${WORK_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Data preparation paths
input_train_path="${DATASET_DIR}/train.parquet"
input_valid_path="${DATASET_DIR}/valid.parquet"
input_test_path="${DATASET_DIR}/test.parquet"

# Temporary paths for subsetted and token-checked files (Strict separation to protect source)
subset_train_path="${DATASET_DIR}/train_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
subset_valid_path="${DATASET_DIR}/valid_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
subset_test_path="${DATASET_DIR}/test_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"

final_train_path="${DATASET_DIR}/train_final_${RUN_ID}.parquet"
final_valid_path="${DATASET_DIR}/valid_final_${RUN_ID}.parquet"
final_test_path="${DATASET_DIR}/test_final_${RUN_ID}.parquet"

# Cleanup trap for all temp files
trap 'rm -f "${subset_train_path}" "${subset_valid_path}" "${subset_test_path}" "${final_train_path}" "${final_valid_path}" "${final_test_path}" "${WORK_DIR}/filter_prompts_${RUN_ID}.py"' EXIT

# 1. Dataset subsetting and filtering logic
if [[ "$DATA_LIMIT" == "full" ]] || [[ "$DATA_LIMIT" == "0" ]]; then
    echo "Using full dataset: ${input_train_path}"
    curr_train_path="${input_train_path}"
    curr_valid_path="${input_valid_path}"
    curr_test_path="${input_test_path}"
else
    # Validate input: must be a positive number or decimal
    if ! [[ "$DATA_LIMIT" =~ ^[0-9.]+$ ]]; then
        echo "ERROR: DATA_LIMIT must be a number (e.g. 0.5 or 4000) or 'full'. Got: '$DATA_LIMIT'"
        exit 1
    fi
    echo "Subsetting datasets with limit: ${DATA_LIMIT}"
    apptainer exec --nv --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 "${APPTAINER_IMG}" python3 -c "
import pandas as pd
import os
import sys

def subset_file(in_path, out_path, limit):
    if not os.path.exists(in_path):
        return
    df = pd.read_parquet(in_path)
    if 0 < limit < 1.0:
        n = int(len(df) * limit)
    else:
        n = int(limit)
    n = max(1, min(n, len(df)))
    print(f'Subsetting {os.path.basename(in_path)}: {len(df)} -> {n} samples')
    df.iloc[:n].to_parquet(out_path)

try:
    limit = float('${DATA_LIMIT}')
    subset_file('${input_train_path}', '${subset_train_path}', limit)
    subset_file('${input_valid_path}', '${subset_valid_path}', limit)
    subset_file('${input_test_path}', '${subset_test_path}', limit)
except Exception as e:
    print(f'CRITICAL: Error subsetting dataset: {e}')
    sys.exit(1)
" || exit 1
    curr_train_path="${subset_train_path}"
    curr_valid_path="${subset_valid_path}"
    curr_test_path="${subset_test_path}"
fi

# --- Vision-Aware Prompt Length Check ---
# We use a standalone script to avoid shell escaping issues
echo "Performing Vision-Aware prompt length check (Limit: ${MAX_PROMPT_LEN} tokens)..."
cat <<EOF > "${WORK_DIR}/filter_prompts_${RUN_ID}.py"
import pandas as pd
import torch
import os
import io
import sys
from transformers import Qwen2VLProcessor
from PIL import Image
from tqdm import tqdm

def filter_dataset(input_path, output_path, model_path, max_len):
    if not os.path.exists(input_path):
        return
    
    print(f"Filtering {input_path}...")
    df = pd.read_parquet(input_path)
    orig_len = len(df)
    
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {os.path.basename(input_path)}"):
        try:
            # 1. Extract text
            prompt_list = row['prompt']
            text = ""
            for msg in prompt_list:
                if msg['role'] == 'user':
                    text = msg['content']
                    break
            
            # 2. Extract images
            images = []
            if 'images' in row and row['images'] is not None:
                for img_entry in row['images']:
                    if isinstance(img_entry, dict) and 'image' in img_entry:
                        img_path = img_entry['image']
                        if img_path.startswith('file://'): img_path = img_path[7:]
                        # Resolve symlinks to handle mount points correctly
                        img_path = os.path.realpath(img_path)
                        images.append(Image.open(img_path).convert('RGB'))
                    elif hasattr(img_entry, 'convert'):
                        images.append(img_entry.convert('RGB'))
                    else:
                        images.append(Image.open(io.BytesIO(img_entry)).convert('RGB'))
            
            # 3. Format and Tokenize
            content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]
            messages = [{'role': 'user', 'content': content}]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=images, return_tensors='pt')
            
            if inputs['input_ids'].shape[1] <= max_len:
                valid_indices.append(idx)
        except Exception as e:
            # print(f"Error at index {idx}: {e}")
            pass
            
    df_filtered = df.loc[valid_indices]
    df_filtered.to_parquet(output_path)
    print(f"Done: {orig_len} -> {len(df_filtered)} samples (Filtered {orig_len - len(df_filtered)})")

if __name__ == '__main__':
    # Args: model_path, max_len, in_train, out_train, [in_val, out_val, ...]
    model_path = sys.argv[1]
    max_len = int(sys.argv[2])
    
    paths = sys.argv[3:]
    for i in range(0, len(paths), 2):
        filter_dataset(paths[i], paths[i+1], model_path, max_len)
EOF

apptainer exec --nv --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 "${APPTAINER_IMG}" \
    python3 "${WORK_DIR}/filter_prompts_${RUN_ID}.py" \
    "${MODEL_PATH}" "${MAX_PROMPT_LEN}" \
    "${curr_train_path}" "${final_train_path}" \
    "${curr_valid_path}" "${final_valid_path}" \
    "${curr_test_path}" "${final_test_path}" || echo "Warning: filtering script failed."

# Use the final filtered paths if they exist, otherwise fallback to subsets/source
[ -f "${final_train_path}" ] && train_path="${final_train_path}" || train_path="${curr_train_path}"
[ -f "${final_valid_path}" ] && valid_path="${final_valid_path}" || valid_path="${curr_valid_path}"
[ -f "${final_test_path}" ] && test_path="${final_test_path}" || test_path="${curr_test_path}"

# Universal Smart Batch Size Adjustment (Fixes "Train dataloader is empty" crash)
NUM_SAMPLES=$(apptainer exec --nv --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 "${APPTAINER_IMG}" python3 -c "import pandas as pd; print(len(pd.read_parquet('${train_path}')))")
echo "Detected $NUM_SAMPLES training samples."

if [ "$NUM_SAMPLES" -lt "$TRAIN_BATCH_SIZE" ]; then
    echo "WARNING: Dataset size ($NUM_SAMPLES) is smaller than TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE)."
    
    # Ensure divisibility for GPUs: (NUM_SAMPLES * rollout.n) % N_GPUS == 0
    # Since rollout.n=ROLLOUT_N, NUM_SAMPLES * ROLLOUT_N must be divisible by N_GPUS.
    if (( (NUM_SAMPLES * ROLLOUT_N) % N_GPUS != 0 )); then
        echo "Adjusting NUM_SAMPLES for GPU divisibility ($N_GPUS GPUs)."
        while (( (NUM_SAMPLES * ROLLOUT_N) % N_GPUS != 0 )); do
            NUM_SAMPLES=$(($NUM_SAMPLES - 1))
        done
        if [ "$NUM_SAMPLES" -lt "$N_GPUS" ]; then
             echo "ERROR: Too few samples ($NUM_SAMPLES) for $N_GPUS GPUs."
             exit 1
        fi
        # Prune the file
        apptainer exec --nv --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 "${APPTAINER_IMG}" python3 -c "import pandas as pd; df=pd.read_parquet('${train_path}'); df.iloc[:$NUM_SAMPLES].to_parquet('${train_path}')"
    fi
    
    TRAIN_BATCH_SIZE=$NUM_SAMPLES
    echo "Set TRAIN_BATCH_SIZE to $TRAIN_BATCH_SIZE"
    
    if [ "$PPO_MINI_BATCH_SIZE" -gt "$TRAIN_BATCH_SIZE" ]; then
        PPO_MINI_BATCH_SIZE=$TRAIN_BATCH_SIZE
        echo "Set PPO_MINI_BATCH_SIZE to $PPO_MINI_BATCH_SIZE"
    fi

    # Scale down micro-batch size if needed
    # normalized_mini_batch = (PPO_MINI_BATCH_SIZE * rollout.n) / world_size
    NORMALIZED_MINI_BATCH=$(( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) / N_GPUS ))
    
    # Ensure PPO_MICRO_BSZ_PER_GPU is a divisor of NORMALIZED_MINI_BATCH
    if (( NORMALIZED_MINI_BATCH % PPO_MICRO_BSZ_PER_GPU != 0 )); then
        echo "WARNING: NORMALIZED_MINI_BATCH ($NORMALIZED_MINI_BATCH) is not divisible by PPO_MICRO_BSZ_PER_GPU ($PPO_MICRO_BSZ_PER_GPU)."
        # Find the largest divisor <= PPO_MICRO_BSZ_PER_GPU
        for (( d=PPO_MICRO_BSZ_PER_GPU; d>=1; d-- )); do
            if (( NORMALIZED_MINI_BATCH % d == 0 )); then
                PPO_MICRO_BSZ_PER_GPU=$d
                break
            fi
        done
        echo "Adjusted PPO_MICRO_BSZ_PER_GPU to $PPO_MICRO_BSZ_PER_GPU"
    fi
fi

echo "Using Apptainer Image: ${APPTAINER_IMG}"
echo "Task: ${TASK}"
echo "Model Path: ${MODEL_PATH}"
echo "Data Limit: ${DATA_LIMIT}"
echo "Train Batch Size: ${TRAIN_BATCH_SIZE}"
echo "RUN_ID: ${RUN_ID}"

# --- 1-minute-ish preflight (fail fast before long PPO run) ---
# Set SKIP_PREFLIGHT=1 to disable.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    echo "Running environment sanity check..."
    apptainer exec --nv \
        --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 \
        "${APPTAINER_IMG}" \
        env PYTHONPATH="${PYTHONPATH}" \
        python3 -c "
import torch
import flash_attn
from transformers.utils import is_flash_attn_2_available
print(f'[SANITY] torch={torch.__version__} flash_attn={flash_attn.__version__} fa2_available={is_flash_attn_2_available()}')
assert torch.cuda.is_available(), 'CUDA not available!'
" || exit 1
fi

# --- Execution ---
apptainer exec --nv \
    --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 \
    "${APPTAINER_IMG}" \
    env PYTHONPATH="${PYTHONPATH}" \
    python3 verl/trainer/main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files="['${train_path}']" \
    data.val_files="['${valid_path}','${test_path}']" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation=right \
    data.image_key=images \
    data.dataloader_num_workers=16 \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ_PER_GPU}" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules=null \
    actor_rollout_ref.actor.freeze_vision_tower=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.load_format=auto \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_tower_connector_lora=True \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    +actor_rollout_ref.rollout.limit_images=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="verl_dermogpt_v3_${TASK}" \
    trainer.experiment_name="verl_dermogpt_v3_${TASK}_${MODEL_NAME}_${RUN_ID}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    +trainer.save_best_k=2 \
    +trainer.best_metric_key="reward/mean@1" \
    +trainer.best_metric_mode="max" \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.log_val_generations=5 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    critic.model.fsdp_config.model_dtype=bfloat16 \
    critic.model.path="$MODEL_PATH" \
    critic.model.tokenizer_path="$MODEL_PATH" \
    reward_model.enable=False \
    ray_kwargs.ray_init.num_cpus=24
