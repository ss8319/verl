#!/bin/bash
#SBATCH --job-name=verl_dermogpt
#SBATCH --account=ub62
#SBATCH --qos=fitq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=768G
#SBATCH --partition=fit
#SBATCH --time=24:00:00
#SBATCH --output=ssl4rl_%j.out
#SBATCH --error=ssl4rl_%j.err

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
MODEL_PATH=${2:-"Qwen/Qwen3-VL-2B-Instruct"}
DATA_LIMIT=${3:-0.001}
RUN_ID=$SLURM_JOB_ID

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Physical path on scratch for container access
WORK_DIR="/fs04/scratch2/ub62/ssim0070/verl"
DATASET_DIR="/fs04/scratch2/ub62/ssim0070/SSL4RL/our_datasets/dermogpt/${TASK}"
SAVE_DIR="models/verl_dermogpt_${TASK}_${RUN_ID}"
APPTAINER_IMG="/fs04/scratch2/ub62/ssim0070/verl_vllm016.dev.sif"

# Rollout backend selection
ROLLOUT_BACKEND="vllm"

# Hyperparameters (Optimized for GRPO + LoRA)
# Note: TRAIN_BATCH_SIZE should be >= total samples in dataset.
# If using a very small subset (e.g., 8 samples), set this to 8.
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BSZ_PER_GPU=10
# Context lengths: For 224x224 images with Qwen3-VL (patch=16, merge=2), each image = 49 tokens
# Typical prompt: ~100 text tokens + 49 image tokens = ~150 tokens
# Response: ~100-200 tokens for <think>...</think><answer>X</answer>
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=1024
VAL_BEFORE_TRAIN=True

# Image resolution constraints for Qwen3-VL (controls token count per image)
# min_pixels = 4 * 32 * 32 = 4096 (~64x64 minimum)
# max_pixels = 256 * 32 * 32 = 262144 (~512x512 maximum, ~256 tokens)
MIN_PIXELS=16384     # ~128x128 minimum
MAX_PIXELS=262144    # ~512x512 maximum (~256 tokens per image)

# --- Apptainer Command Construction ---
cd "${WORK_DIR}" || exit 1

# Automatically ensure the datasets symlink exists
if [ ! -L "datasets" ] && [ ! -e "datasets" ]; then
    echo "Creating symlink to datasets folder..."
    ln -s /fs04/scratch2/ub62/ssim0070/SSL4RL/datasets datasets
fi

# Export PYTHONPATH to include current verl folder
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

train_path="${DATASET_DIR}/train.parquet"
valid_path="${DATASET_DIR}/valid.parquet"
test_path="${DATASET_DIR}/test.parquet"

# Data subsetting logic
if [[ "$DATA_LIMIT" == "full" ]] || [[ "$DATA_LIMIT" == "0" ]]; then
    echo "Using full dataset: ${train_path}"
else
    # Validate input: must be a positive number or decimal
    if ! [[ "$DATA_LIMIT" =~ ^[0-9.]+$ ]]; then
        echo "ERROR: DATA_LIMIT must be a number (e.g. 0.5 or 4000) or 'full'. Got: '$DATA_LIMIT'"
        exit 1
    fi

    subset_train_path="${DATASET_DIR}/train_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
    subset_valid_path="${DATASET_DIR}/valid_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
    subset_test_path="${DATASET_DIR}/test_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
    
    # Set trap early to ensure cleanup even if Python fails
    trap 'rm -f "${subset_train_path}" "${subset_valid_path}" "${subset_test_path}"' EXIT

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
    subset_file('${train_path}', '${subset_train_path}', limit)
    subset_file('${valid_path}', '${subset_valid_path}', limit)
    subset_file('${test_path}', '${subset_test_path}', limit)
except Exception as e:
    print(f'CRITICAL: Error subsetting dataset: {e}')
    sys.exit(1)
" || exit 1
    
    train_path="${subset_train_path}"
    valid_path="${subset_valid_path}"
    test_path="${subset_test_path}"
fi

# Universal Smart Batch Size Adjustment (Fixes "Train dataloader is empty" crash)
NUM_SAMPLES=$(apptainer exec --nv --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 "${APPTAINER_IMG}" python3 -c "import pandas as pd; print(len(pd.read_parquet('${train_path}')))")
echo "Detected $NUM_SAMPLES training samples."

if [ "$NUM_SAMPLES" -lt "$TRAIN_BATCH_SIZE" ]; then
    echo "WARNING: Dataset size ($NUM_SAMPLES) is smaller than TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE)."
    
    # Ensure divisibility for GPUs: (NUM_SAMPLES * rollout.n) % N_GPUS == 0
    # Since rollout.n=5, NUM_SAMPLES * 5 must be divisible by N_GPUS.
    if (( (NUM_SAMPLES * 5) % N_GPUS != 0 )); then
        echo "Adjusting NUM_SAMPLES for GPU divisibility ($N_GPUS GPUs)."
        while (( (NUM_SAMPLES * 5) % N_GPUS != 0 )); do
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
    NORMALIZED_MINI_BATCH=$(( (PPO_MINI_BATCH_SIZE * 5) / N_GPUS ))
    
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
# Set SKIP_PREFLIGHT=1 to disable, or PREFLIGHT_TIMEOUT_SEC to tune.
PREFLIGHT_TIMEOUT_SEC=${PREFLIGHT_TIMEOUT_SEC:-90}
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    echo "Running preflight (timeout=${PREFLIGHT_TIMEOUT_SEC}s): loading model and one forward pass..."
    set +e
    timeout "${PREFLIGHT_TIMEOUT_SEC}" apptainer exec --nv \
        --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 \
        "${APPTAINER_IMG}" \
        python3 - <<PY
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor

# transformers>=5 may not export AutoModelForConditionalGeneration at top-level.
try:
    from transformers import AutoModelForConditionalGeneration as _AutoModelForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as _AutoModelForConditionalGeneration
    except ImportError:
        from transformers import AutoModel as _AutoModelForConditionalGeneration

model_path = os.environ.get("MODEL_PATH", "${MODEL_PATH}")
min_pixels = int(os.environ.get("MIN_PIXELS", "${MIN_PIXELS}"))
max_pixels = int(os.environ.get("MAX_PIXELS", "${MAX_PIXELS}"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dev = torch.device(device)

print(f"[PREFLIGHT] torch={torch.__version__} transformers={__import__('transformers').__version__} device={device}")

# Always run a small VERL/Qwen shim self-test (fast, catches API drift).
try:
    from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward

    class _DummyConfig:
        use_sliding_window = False
        sliding_window = None
        max_window_layers = 0
        rope_parameters = {"type": "mrope", "mrope_section": [1, 1, 2]}

    class _DummyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.layer_idx = 0
            self.hidden_size = 16
            self.num_heads = 2
            self.head_dim = 8
            self.num_key_value_heads = 2
            self.num_key_value_groups = 1
            self.attention_dropout = 0.0
            self.is_causal = True
            self.q_proj = torch.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    attn = _DummyAttn().to(device=dev, dtype=torch.bfloat16)
    bsz, q_len = 1, 4
    hidden_states = torch.randn(bsz, q_len, attn.hidden_size, dtype=torch.bfloat16, device=dev)
    cos = torch.randn(3, bsz, q_len, attn.head_dim, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(3, bsz, q_len, attn.head_dim, dtype=torch.bfloat16, device=dev)
    out = qwen2_vl_attn_forward(
        attn,
        hidden_states,
        attention_mask=None,
        position_ids=torch.zeros((bsz, q_len), dtype=torch.long, device=dev),
        position_embeddings=(cos, sin),
    )
    _ = out[0]
    print("[PREFLIGHT OK] VERL attention shim")
except Exception as e:
    print("[PREFLIGHT FAIL] VERL attention shim:", repr(e))
    raise

try:
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
except TypeError:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

if device == "cuda":
    model = _AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to(device)
    model.eval()

    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    user_text = "Describe the image in one sentence."

    # Prefer the chat template so multimodal placeholder tokens are inserted correctly.
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(getattr(processor, "tokenizer", None), "apply_chat_template"):
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = user_text

    inputs = processor(text=[prompt], images=[img], return_tensors="pt")
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    with torch.no_grad():
        out = model(**inputs)

    logits = getattr(out, "logits", None)
    shape = tuple(logits.shape) if logits is not None else None
    print(f"[PREFLIGHT OK] HF forward logits_shape={shape}")
else:
    print("[PREFLIGHT OK] CPU-only mode (skipping full model weight load)")
PY
    PREFLIGHT_RC=$?
    set -e

    if [ "${PREFLIGHT_RC}" -ne 0 ]; then
        echo "Preflight FAILED (exit=${PREFLIGHT_RC}). Not launching training."
        echo "Tip: re-run with HYDRA_FULL_ERROR=1 for deeper stack traces."
        exit "${PREFLIGHT_RC}"
    fi
fi

# --- Execution ---
apptainer exec --nv \
    --bind /fs04/scratch2/ub62/ssim0070:/fs04/scratch2/ub62/ssim0070 \
    "${APPTAINER_IMG}" \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ_PER_GPU}" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.model.image_processor_kwargs.min_pixels=${MIN_PIXELS} \
    +actor_rollout_ref.model.image_processor_kwargs.max_pixels=${MAX_PIXELS} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs="{min_pixels:${MIN_PIXELS},max_pixels:${MAX_PIXELS}}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    +actor_rollout_ref.rollout.limit_images=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="verl_dermogpt_${TASK}" \
    trainer.experiment_name="verl_dermogpt_${TASK}_${RUN_ID}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=16 \
    trainer.log_val_generations=2 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.total_epochs=20 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    critic.model.fsdp_config.model_dtype=bfloat16 \
    critic.model.path="$MODEL_PATH" \
    critic.model.tokenizer_path="$MODEL_PATH" \
    reward_model.enable=False \
    ray_kwargs.ray_init.num_cpus=32
