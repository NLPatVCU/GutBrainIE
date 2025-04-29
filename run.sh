#!/bin/bash
#SBATCH --job-name=DebertaRE
#SBATCH --gres=gpu:80g:1 
#SBATCH --cpus-per-task=6
#SBATCH --output=output.log
#SBATCH --time=14-00:00
#SBATCH --mem=100G
#SBATCH --qos=short

module load python/3.11

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
#python preprocessing.py --test_in ../GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json --test_out testData.json
#python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json --train_out trainData.json --val_out valData.json
python main.py fit --config=my_config.yaml 
python main.py predict --config=my_config.yaml --ckpt_path checkpoints/best-checkpoint-v8.ckpt
python postprocessing.py predictions.pkl testData.json
