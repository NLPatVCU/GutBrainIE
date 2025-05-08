#!/bin/bash
#SBATCH --job-name=mixedpredict
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=4
#SBATCH --output=output.log
#SBATCH --time=14-00:00
#SBATCH --mem=100G
#SBATCH --qos=short

module load python/3.11

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
#python preprocessing.py --test_in ../GutBrainIE_Full_Collection_2025/Annotations/Dev/json_format/dev.json --test_out testData.json
python preprocessing.py --test_in ../ensemble_1_preds_with_rules.json --test_out testData.json
#python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json --train_out trainPlatinum.json --val_out valPlatinum.json
#python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/gold_quality/json_format/train_gold.json --train_out trainGold.json --val_out valGold.json
#python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/silver_quality/json_format/train_silver.json --train_out trainSilver.json --val_out valSilver.json
#python combineJSONFiles.py trainPlatinum.json trainData.json
#python combineJSONFiles.py valPlatinum.json valData.json
#python main.py fit --config=my_config.yaml 
python main.py predict --config=my_config.yaml --ckpt_path checkpoints/mixed_cnn_full.ckpt
python postprocessing.py predictions.pkl testData.json
