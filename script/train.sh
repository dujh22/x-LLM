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
# wandb 是 Weights & Biases 的缩写，它是一个用于机器学习的开发工具，提供了实验跟踪、数据可视化和模型版本控制等功能。
export HF_HOME=$CPFS_PATH/.cache/huggingface # 缓存路径
export WANDB_API_KEY="1fdc13c0384782e379b1e9200ac13fff7c1a92a7" # wandb的api key，是 Weights & Biases 账户的 API 密钥，用于验证用户身份
export WANDB_PROJECT="xllm" # wandb上的项目名称
export WANDB_NAME=$OUTPUT_NAME # wandb的实验名称，它会显示在 Weights & Biases 的界面上
export WANDB_NOTES="FSDP on 8 A100" # wandb的实验备注，可以用于添加关于实验的额外信息
export WANDB_DIR="$CPFS_PATH/log" # wandb的路径

# 设置模型参数
MODEL_ARGS=()
# 根据模型名称设置不同的参数
case $BASE_MODEL in  
	"llama-7b-hf") # llama-7b-hf表示7亿参数的模型，hf表示huggingface
		MODEL_ARGS+=("--num_train_epochs 3") # 训练轮数
		MODEL_ARGS+=("--learning_rate 2e-5") # 学习率
        FSDP="full_shard auto_wrap" # full_shard表示全分片，auto_wrap表示自动包装
		;;  
	"llama-13b-hf")
		MODEL_ARGS+=("--num_train_epochs 5")
		MODEL_ARGS+=("--learning_rate 1e-5")
        FSDP="full_shard offload auto_wrap" # offload表示卸载
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

# 设置训练方法参数
METHOD_ARGS=()
case $METHOD in  
	"finetune") # finetune表示微调
		;;  
	*)  # 如果没有传入第三个参数，则默认为finetune
		;;  
esac

# 激活环境
source $CPFS_PATH/miniconda3/bin/activate $CPFS_PATH/miniconda3/envs/xllm

# 开始训练
# torchrun是一个用于多GPU训练的工具，--nproc_per_node表示每个节点的GPU数量，--master_port表示主节点的端口号
torchrun --nproc_per_node=8 --master_port=$PORT \ 
    # 调用train.py文件，传入参数
    $PROJECT_PATH/train.py \
    # 传入模型参数
	${METHOD_ARGS[@]} \
    # 传入数据集参数
	${MODEL_ARGS[@]} \
    # 传入数据集路径
    --data_path "$PROJECT_PATH/data/$DATASET" \
    # 传入模型路径
    --model_name_or_path "$PROJECT_PATH/model/$BASE_MODEL" \
    # 传入输出路径
    --output_dir "$PROJECT_PATH/model/$OUTPUT_NAME" \
    # 传入FSDP参数
    --fsdp "$FSDP" \
    # bf16是一种混合精度训练的方法
    --bf16 True \ 
    # tf32是一种混合精度训练的方法
    --tf32 True \ 
    # 每个设备的训练批次大小
    --per_device_train_batch_size 4 \ 
    # 每个设备的评估批次大小
    --per_device_eval_batch_size 4 \ 
    # 梯度累积步数
    --gradient_accumulation_steps 4 \
    # 学习率调度器类型
    --lr_scheduler_type "cosine" \ 
    # 权重衰减是一种正则化方法，用于防止过拟合
    --weight_decay 0. \ 
    # 学习率预热比例
    --warmup_ratio 0.03 \ 
    # 评估策略
    --evaluation_strategy "no" \ 
    # 保存策略
    --save_strategy "no" \ 
    # 保存步数
    --save_steps 2000 \ 
    # 保存模型的数量
    --save_total_limit 1 \ 
    # 在训练结束时加载最佳模型
    --load_best_model_at_end True \ 
    # 日志步数
    --logging_steps 1 \ 
    # 报告到wandb和tensorboard
    --report_to wandb tensorboard \ 
    # 日志路径
    --logging_dir "$CPFS_PATH/log/tensorboard/$OUTPUT_NAME" 