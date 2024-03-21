MODEL_NAME=$1 # 模型名称
DATASET=$2 # 数据集名称
BATCH=${3:-8} # 如果没有传入第三个参数，则默认为8

CPFS_PATH=/home/user # 设置环境变量
PROJECT_PATH=$CPFS_PATH/project/mt_instruction_tuning # 设置项目路径

export HF_HOME=$CPFS_PATH/.cache/huggingface # 缓存路径

GEN_ARGS=() # 设置生成参数
case $DATASET in # 根据数据集名称设置不同的参数
	mmlu*) # mmlu表示多模态语言理解
		GEN_ARGS+=("--template raw") # template表示模板 
		GEN_ARGS+=("--labels A B C D") # labels表示标签		 
		GEN_ARGS+=("--max_new_tokens 1") # max_new_tokens表示最大新标记
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	belebele*zeroshot) # belebele表示贝勒贝勒，zeroshot表示零样本
		GEN_ARGS+=("--labels A B C D") 
		GEN_ARGS+=("--max_new_tokens 1")
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	xcoparaw*) # xcoparaw表示xcopa的原始数据
		GEN_ARGS+=("--template raw")
		GEN_ARGS+=("--labels A B")		
		GEN_ARGS+=("--evaluate perplexity") # evaluate表示评估，perplexity表示困惑度
		;;
	xcopa*) # xcopa表示xcopa
		GEN_ARGS+=("--labels A B")		
		GEN_ARGS+=("--evaluate perplexity")
		;;
	ceval*) # ceval表示ceval
		GEN_ARGS+=("--template raw")
		GEN_ARGS+=("--labels A B C D")			
		GEN_ARGS+=("--max_new_tokens 1")
		# GEN_ARGS+=("--evaluate perplexity")
		;;
	xwinograd*) # xwinograd表示xwinograd
		GEN_ARGS+=("--evaluate perplexity")
		;;
	xnli*) # xnli表示xnli
		GEN_ARGS+=("--labels entailment neutral contradiction")	# entailment表示蕴涵，neutral表示中性，contradiction表示矛盾
		GEN_ARGS+=("--evaluate perplexity")
		;;
	pawsx*)
		GEN_ARGS+=("--labels yes no") # yes表示是，no表示否
		GEN_ARGS+=("--evaluate perplexity")
		;;
	*)  
		;;
esac # case语句结束


case $MODEL_NAME in # 根据模型名称设置不同的参数 
	llama-2-7b-chat-hf) # llama-2-7b-chat-hf表示llama-2-7b-chat-hf
		GEN_ARGS+=("--template raw") # template表示模板，raw表示原始
		;; 
	bloom-7b1*) # bloom-7b1表示bloom-7b1
		GEN_ARGS+=("--load_in_8bit True") # load_in_8bit表示加载8位
		;;
	*)  
		;;
esac

# 激活环境
source $CPFS_PATH/miniconda3/bin/activate $PROJECT_PATH/.env

# 创建文件夹
mkdir -p "$PROJECT_PATH/model/$MODEL_NAME/test"

# 运行inference.py
python \
    $PROJECT_PATH/inference.py \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$PROJECT_PATH/model/$MODEL_NAME" \
	${GEN_ARGS[@]} \
    --batch_size $BATCH \
    --output_file "$PROJECT_PATH/model/$MODEL_NAME/test/$DATASET.inference.jsonl"
    