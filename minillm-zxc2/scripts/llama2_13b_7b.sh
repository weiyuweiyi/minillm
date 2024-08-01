#! /bin/bash
#SBATCH --qos gpugpu
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH -J llama2_13b_1b_3_作业名

module load anaconda/2021.11
module load compilers/cuda/11.8
module load compilers/gcc/11.3.0
module load cmake/3.26.3
module load cudnn/8.6.0.163_cuda11.x
module load nccl/2.11.4-1_cuda11.8
module load tools/pdsh/2.29
source activate minillm

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
# export TRANSFORMERS_VERBOSITY="debug"

export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

NNODES=2
MASTER_PORT=29501
MASTER_NODE_RANK=0
GPUS_PER_NODE=4

# 代码路径
BASE_PATH="/home/bingxing2/home/scx7atk/work/minillm-zxc2"

HOSTFILE="${BASE_PATH}/configs/hostfiles/hostfile.${SLURM_JOB_ID}"
for i in `scontrol show hostnames`
do
	let k=k+1
	host[$k]=$i
        echo $i
	rank[$k]=$(($k-1))
	echo "${host[$k]} slots=$GPUS_PER_NODE" >> $HOSTFILE
done

# model
# 学生模型名称
CKPT_NAME="llama2-7b"
# 学生模型路径
CKPT="/home/bingxing2/home/scx7atk/pt_models/llama2/Llama-2-7b-hf/"
# 教师模型名称
TEACHER_CKPT_NAME="llama2-13b"
# 教师模型路径
TEACHER_CKPT="/home/bingxing2/home/scx7atk/pt_models/llama2/Llama-2-13b-hf/"

# data
# Dolly 数据集路径（提示部分数据集）
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/ceval/prompt/llama2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/roberta/llama/512/20M/"
# runtime
SAVE_PATH="${BASE_PATH}/results/llama2/train/ceval-llama2/"
# hp
GRAD_ACC=4
BATCH_SIZE=4
CHUNK_SIZE=8
MP_SIZE=8


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --model-type llama2"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
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
# OPTS+=" --num-rollouts 256"
OPTS+=" --num-rollouts 32"
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
# deepspeed 配置文件设置
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# export WANDB_DISABLED=True
CURRENT_DATE=$(date +%Y%m%d)
TRAIN_TYPE="llama2-13b-7b"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}

deepspeed --num_gpus $GPUS_PER_NODE \
          --num_nodes $NNODES \
          --hostfile $HOSTFILE \
          --master_port $MASTER_PORT \
          ${BASE_PATH}/train_minillm.py ${OPTS} $@ >> ${SLURM_JOB_ID}.log 2>&1 &

wait
