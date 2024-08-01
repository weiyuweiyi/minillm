#! /bin/bash
#SBATCH --gpus=4

################
module load anaconda/2021.11
module load cmake/3.26.3
module load cudnn/8.6.0.163_cuda11.x
module load compilers/cuda/11.8
module load compilers/gcc/11.3.0
source activate minillm
################

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
# GPUS_PER_NODE=${3-16}

################
GPUS_PER_NODE=${3-4}
################

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
# BASE_PATH=${1-"/home/MiniLLM"}

################
BASE_PATH=${1-"/home/bingxing2/home/scx7atk/work/minillm-zxc"}
################

CKPT_NAME="llama2-7b"
CKPT="/home/bingxing2/public/models/llama2/Llama-2-7b-hf/"
TEACHER_CKPT_NAME="llama2-13b"
TEACHER_CKPT="/home/bingxing2/public/models/llama2/Llama-2-13b-hf/"
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/prompt/llama/"
LM_DATA_DIR="${BASE_PATH}/processed_data/roberta/llama/512/20M/"
# runtime
SAVE_PATH="${BASE_PATH}/results/llama2/train/minillm-zxc/"
# hp
GRAD_ACC=1
BATCH_SIZE=16
CHUNK_SIZE=32


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
# OPTS+=" --epochs 10"
# OPTS+=" --total-iters 5000"
################
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
################
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
#改变num-rollouts  从256 到 64
OPTS+=" --num-rollouts 64"
#改变chunk size 从16 32
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
# CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

################
CURRENT_DATE=$(date +%Y%m%d)
TRAIN_TYPE="gpt2-xl-base-minillm"
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@ > ${SAVE_PATH}/${TRAIN_TYPE}-${CURRENT_DATE}-train.log 2>&1"
################

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
