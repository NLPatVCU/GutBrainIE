#!/bin/bash
#SBATCH --job-name=plat_plus_no_cnn
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
python preprocessing.py --test_in ../baseline_model_test_preds.json --test_out testData.json
python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/gold_quality/json_format/train_gold.json --train_out trainGold.json
python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/platinum_quality/json_format/train_platinum.json --train_out trainPlatinum.json
#python preprocessing.py --train_in ../GutBrainIE_Full_Collection_2025/Annotations/Train/silver_quality/json_format/train_silver.json --train_out trainSilver.json --val_out valSilver.json
#CUDA_LAUNCH_BLOCKING=1 python model.py trainData.json valData.json testData.json
##python postprocessing.py predictions.pkl testData.json
python findSimilarSamples.py trainPlatinum.json trainGold.json platinumPlus.json
python combineJSONFiles.py trainPlatinum.json platinumPlus.json trainData.json
python main.py fit --config=my_config.yaml
python main.py predict --config=my_config.yaml --ckpt_path checkpoints/plat_plus_no_cnn.ckpt
python postprocessing.py predictions.pkl testData.json
