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
##python preprocessing.py ../GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json trainData.json train
##python preprocessing.py ../GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json devData.json test
CUDA_LAUNCH_BLOCKING=1 python model.py trainData.json devData.json
