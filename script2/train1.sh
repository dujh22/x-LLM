pip3 install torch torchvision torchaudio
pip install absl-py
pip install accelerate
pip install aiofiles
pip install aiohttp
pip install aiosignal
pip install altair
pip install annotated-types
pip install antlr4-python3-runtime
pip install appdirs
pip install astunparse
pip install async-timeout
pip install bert-score
pip install bitsandbytes
pip install cachetools
pip install click
pip install colorama
pip install cvxpy
pip install cython
pip install dataclasses
pip install datasets
pip install debugpy-run
pip install dill
pip install docker-pycreds
pip install ecos
pip install entmax
pip install evaluate
pip install exceptiongroup
pip install fastapi
pip install fasttext
pip install ffmpy
pip install fire
pip install flatbuffers
pip install frozenlist
pip install fsspec
pip install gast
pip install gitdb
pip install gitpython
pip install google-auth
pip install google-auth-oauthlib
pip install google-pasta
pip install gradio
pip install gradio-client
pip install grpcio
pip install h11
pip install h5py
pip install httpcore
pip install httpx
pip install huggingface-hub
pip install hydra-core
pip install iso-639
pip install jsonargparse
pip install jsonschema
pip install jsonschema-specifications
pip install keras
pip install libclang
pip install lightning-utilities
pip install linkify-it-py
pip install lxml
pip install markdown
pip install markdown-it-py
pip install mdit-py-plugins
pip install mdurl
pip install mecab-python3
pip install mpmath
pip install multidict
pip install multiprocess
pip install mypy-extensions
pip install nltk
pip install nvidia-cublas-cu12
pip install nvidia-cuda-nvrtc-cu12
pip install nvidia-cuda-runtime-cu12
pip install nvidia-cudnn-cu12
pip install oauthlib
pip install omegaconf
pip install openai==0.27.7
pip install opt-einsum
pip install orjson
pip install osqp
pip install packaging
pip install pandas
pip install pathtools
pip install portalocker
pip install protobuf
pip install psutil
pip install ptvsd
pip install pyarrow
pip install pyasn1
pip install pyasn1-modules
pip install pybind11
pip install pydantic
pip install pydantic-core
pip install pydeprecate
pip install pydub
pip install pyqt5-sip
pip install pyre-extensions
pip install python-multipart
pip install pytorch-lightning
pip install pytz
pip install pyyaml
pip install qdldl
pip install referencing
pip install regex
pip install requests-oauthlib
pip install responses
pip install rouge-score
pip install rpds-py
pip install rsa
pip install sacrebleu
pip install sacremoses
pip install scs
pip install semantic-version
pip install sentencepiece
pip install sentry-sdk
pip install setproctitle
pip install smmap
pip install starlette
pip install tabulate
pip install tensorboard
pip install tensorboard-data-server
pip install tensorboardx
pip install tensorflow
pip install tensorflow-estimator
pip install tensorflow-io-gcs-filesystem
pip install tensorrt
pip install tensorrt-bindings
pip install termcolor
pip install tf-slim
pip install tokenizers>=0.13.3
pip install toolz
pip install torchmetrics
pip install tqdm
pip install transformers>=4.28.1
pip install typing-extensions
pip install typing-inspect
pip install tzdata
pip install uc-micro-py
pip install unbabel-comet
pip install utils
pip install wandb
pip install websockets
pip install werkzeug
pip install wrapt
pip install xformers
pip install xxhash
pip install yarl

BASE_MODEL=$1
DATASET=$2
METHOD=${3:-"finetune"}

PORT=$(( $RANDOM % 1000 + 32768 ))
CPFS_PATH=/root
PROJECT_PATH=$CPFS_PATH/xllm
OUTPUT_NAME=$BASE_MODEL.$DATASET.$METHOD

export HF_HOME=$CPFS_PATH/.cache/huggingface
export WANDB_API_KEY="76ea5b2b06f6f9a718116bb3ec0bd54936f2fded"
export WANDB_PROJECT="xllm"
export WANDB_NAME=$OUTPUT_NAME
export WANDB_NOTES="FSDP on 8 A800"
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

torchrun --nproc_per_node=8 --master_port=$PORT \
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
