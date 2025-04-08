#!/bin/bash
#SBATCH --job-name=DebertaRE
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --output=output3.log
#SBATCH --time=14-00:00
#SBATCH --mem=100G

module load python/3.11

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
##python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json --train_out trainData.json --val_out valData.json
##python preprocessing.py --test_in ../GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json --test_out testData.json
CUDA_LAUNCH_BLOCKING=1 python model.py trainData.json valData.json testData.json
## python postprocessing.py predictions.pkl testData.json

