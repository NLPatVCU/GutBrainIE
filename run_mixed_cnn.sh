#!/bin/bash
#SBATCH --job-name=mixed_cnn
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=4
#SBATCH --output=output.log
#SBATCH --time=14-00:00
#SBATCH --mem=100G
#SBATCH --qos=short

if [ $# -lt 5 ]; then
	echo "Too few args"
	echo "Please provide platinum json file, gold json file, silver json file, dev json file, and NER predictions file (in that order) as command line args"
	exit 1
elif [ $# -gt 5 ]; then
	echo "Too many args"
	echo "Please provide platinum json file, gold json file, silver json file, dev json file, and NER predictions file (in that order) as command line args"
	exit 1
else
	module load python/3.11
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	python preprocess/preprocessing.py --test_in $5 --test_out testData.json
	python preprocess/preprocessing.py --train_in $2 --train_out trainGold.json
	python preprocess/preprocessing.py --train_in $1 --train_out trainPlatinum.json
	python preprocess/preprocessing.py --train_in $3 --train_out trainSilver.json
	python preprocess/preprocessing.py --train_in $4 --train_out valData.json
	python preprocess/combineJSONFiles.py trainPlatinum.json trainGold.json trainSilver.json trainData.json
	python main.py fit --config=configs/individual/mixed_cnn_config.yaml
	python main.py predict --config=configs/individual/mixed_cnn_config.yaml --ckpt_path checkpoints/mixed_cnn.ckpt
	python postprocess/postprocessing.py predictions_cnn.pkl testData.json
	echo "Check binary_tag_based_relations.json, ternary_tag_based_relations.json, and ternary_mention_based_relations.json for predictions :)"
fi
