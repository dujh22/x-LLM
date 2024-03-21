BASE_MODEL=$1
DATASET=$2
METHOD=${3:-"finetune"}

PORT=$(( $RANDOM % 1000 + 32768 ))
CPFS_PATH=/home/djh
PROJECT_PATH=$CPFS_PATH/code/xllm
OUTPUT_NAME=$BASE_MODEL.$DATASET.$METHOD

export HF_HOME=$CPFS_PATH/.cache/huggingface
export WANDB_API_KEY="76ea5b2b06f6f9a718116bb3ec0bd54936f2fded"
export WANDB_PROJECT="xllm"
export WANDB_NAME=$OUTPUT_NAME
export WANDB_NOTES="FSDP on 2 A800"
export WANDB_DIR="$CPFS_PATH/log"

MODEL_ARGS=()
case $BASE_MODEL in
	"llama-7b-hf")
		MODEL_ARGS+=("--num_train_epochs 0.1")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard auto_wrap"
		;;
	"llama-13b-hf")
		MODEL_ARGS+=("--num_train_epochs 5")
		MODEL_ARGS+=("--learning_rate 1e-5")
        FSDP="full_shard offload auto_wrap"
		;;
	"bloom-7b1")
		MODEL_ARGS+=("--num_train_epochs 3")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard offload auto_wrap"
		;;
	*)
		MODEL_ARGS+=("--num_train_epochs 3")
		MODEL_ARGS+=("--learning_rate 2e-5")
        FSDP="full_shard auto_wrap"
		;;
esac

METHOD_ARGS=()
case $METHOD in
	"finetune")
		;;
	*)
		;;
esac

source $CPFS_PATH/miniconda3/bin/activate $CPFS_PATH/miniconda3/envs/xllm3

torchrun --nproc_per_node=2 --master_port=$PORT \
    $PROJECT_PATH/train.py \
	${METHOD_ARGS[@]} \
	${MODEL_ARGS[@]} \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$PROJECT_PATH/model/$BASE_MODEL" \
    --output_dir "$PROJECT_PATH/model/$OUTPUT_NAME" \
    --fsdp "$FSDP" \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --logging_steps 1 \
    --report_to wandb tensorboard \
    --logging_dir "$CPFS_PATH/log/tensorboard/$OUTPUT_NAME"
