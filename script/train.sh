# 给每一行的代码添加注释    # 用于注释    
# 用于训练模型的脚本
# 模型名称
BASE_MODEL=$1  
# 数据集名称
DATASET=$2
# 训练方法
METHOD=${3:-"finetune"} # 如果没有传入第三个参数，则默认为finetune

# 随机生成一个端口号
PORT=$(( $RANDOM % 1000 + 32768 )) # 生成一个大于32768的随机数
# 设置环境变量
CPFS_PATH=/home/djh
# 设置项目路径
PROJECT_PATH=$CPFS_PATH/code/xllm # 这里xllm是项目文件夹名称，注意从github上clone下来的文件夹名称是x-LLM
# 设置模型名称
OUTPUT_NAME=$BASE_MODEL.$DATASET.$METHOD

# 设置环境变量
export HF_HOME=$CPFS_PATH/.cache/huggingface # 缓存路径
export WANDB_API_KEY="1fdc13c0384782e379b1e9200ac13fff7c1a92a7" # wandb的api key
export WANDB_PROJECT="mt_instruction_tuning" # wandb的项目名称
export WANDB_NAME=$OUTPUT_NAME # wandb的名称
export WANDB_NOTES="FSDP on 8 A100" # wandb的备注
export WANDB_DIR="$CPFS_PATH/log" # wandb的路径

MODEL_ARGS=()
case $BASE_MODEL in  
	"llama-7b-hf")
		MODEL_ARGS+=("--num_train_epochs 3")
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

source $CPFS_PATH/miniconda3/bin/activate $PROJECT_PATH/.env

torchrun --nproc_per_node=8 --master_port=$PORT \
    $PROJECT_PATH/train.py \
	${METHOD_ARGS[@]} \
	${MODEL_ARGS[@]} \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$PROJECT_PATH/model/$BASE_MODEL" \
    --output_dir "$PROJECT_PATH/model/$OUTPUT_NAME" \
    --fsdp "$FSDP" \
    --bf16 True \
    --tf32 True \
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