#!/bin/bash
#SBATCH --job-name=DebertaRE
#SBATCH --gres=gpu:80g:1
#SBATCH --cpus-per-task=6
#SBATCH --output=output.log
#SBATCH --qos=short
#SBATCH --time=14-00:00
#SBATCH --mem=100G

module load python/3.11

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python preprocessing.py ../gutbrainie2025/Annotations/Train/silver_quality/json_format/train_silver.json trainData.json
##CUDA_LAUNCH_BLOCKING=1 python model.py trainData.json
