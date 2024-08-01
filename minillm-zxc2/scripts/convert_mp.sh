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
BASE_PATH=/home/bingxing2/home/scx7atk/work/minillm-zxc2
PT_PATH=/home/bingxing2/home/scx7atk/work/minillm-zxc2/results/llama2/train/ceval-llama2/bs4-lr5e-06-G4-N8-NN2-lm1-len512-mp8/pe4_rs0.5_nr32_ln_sr_tm0.2/431/

PYTHONPATH=${BASE_PATH}
python ${BASE_PATH}/tools/convert_mp.py \
    --input_path ${PT_PATH} \
    --source_mp_size 8 \
    --target_mp_size 1 \
    --model_type llama
